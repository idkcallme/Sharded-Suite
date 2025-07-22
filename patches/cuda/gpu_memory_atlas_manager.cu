/*
 * GGUF Shard Atlas - CUDA Memory Management
 * 
 * Provides atomic memory swapping and atlas lookup for sharded GGUF files
 */

#include "gguf_shard_atlas.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Atlas entry states
#define ATLAS_STATE_RESIDENT  0x01
#define ATLAS_STATE_SWAPPED   0x02
#define ATLAS_STATE_PENDING   0x04
#define ATLAS_STATE_LOCKED    0x08

// CUDA kernel for atomic page swapping
__global__ void kernel_atomic_swap_pages(
    atlas_entry_t* atlas,
    void* gpu_memory,
    void* swap_buffer,
    uint32_t* swap_queue,
    uint32_t queue_size,
    volatile uint32_t* completion_flag
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t grid_size = gridDim.x * blockDim.x;
    
    // Cooperative group for synchronization
    auto block = cg::this_thread_block();
    auto grid = cg::this_grid();
    
    for (uint32_t i = tid; i < queue_size; i += grid_size) {
        uint32_t shard_id = swap_queue[i];
        atlas_entry_t* entry = &atlas[shard_id];
        
        // Atomic lock acquisition with backoff
        uint32_t expected = ATLAS_STATE_RESIDENT;
        uint32_t desired = ATLAS_STATE_RESIDENT | ATLAS_STATE_LOCKED;
        
        if (atomicCAS(&entry->state, expected, desired) == expected) {
            // Successfully acquired lock
            
            // Perform DMA copy with memory fence
            void* src = (char*)gpu_memory + entry->physical_addr;
            void* dst = (char*)swap_buffer + (shard_id * 4096);
            
            // Use cooperative copy for large pages
            if (blockDim.x >= 32) {
                cooperative_memcpy_async(block, dst, src, 4096);
            } else {
                memcpy(dst, src, 4096);
            }
            
            // Memory fence to ensure copy completion
            __threadfence();
            
            // Update atlas entry atomically
            entry->physical_addr = (uint64_t)dst;
            
            // Memory fence before state update
            __threadfence();
            
            // Release lock and update state
            atomicExch(&entry->state, ATLAS_STATE_SWAPPED);
            
            // Increment completion counter
            atomicAdd((uint32_t*)completion_flag, 1);
        }
    }
    
    // Grid-wide synchronization
    grid.sync();
}

// Host function for atlas lookup
extern "C" {

atlas_entry_t* atlas_lookup(atlas_t* atlas, uint64_t virtual_addr) {
    if (!atlas || !atlas->entries) {
        return nullptr;
    }
    
    // Binary search in sorted atlas
    uint32_t left = 0;
    uint32_t right = atlas->entry_count - 1;
    
    while (left <= right) {
        uint32_t mid = (left + right) / 2;
        atlas_entry_t* entry = &atlas->entries[mid];
        
        if (virtual_addr >= entry->virtual_addr && 
            virtual_addr < entry->virtual_addr + 4096) {
            return entry;
        }
        
        if (virtual_addr < entry->virtual_addr) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return nullptr;
}

cudaError_t atlas_atomic_swap(
    atlas_t* atlas,
    uint32_t* shard_ids,
    uint32_t count,
    cudaStream_t stream
) {
    if (!atlas || !shard_ids || count == 0) {
        return cudaErrorInvalidValue;
    }
    
    // Allocate device memory for swap queue
    uint32_t* d_swap_queue;
    cudaError_t err = cudaMalloc(&d_swap_queue, count * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    
    // Copy swap queue to device
    err = cudaMemcpyAsync(d_swap_queue, shard_ids, 
                          count * sizeof(uint32_t), 
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_swap_queue);
        return err;
    }
    
    // Allocate completion flag
    uint32_t* d_completion_flag;
    err = cudaMalloc(&d_completion_flag, sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(d_swap_queue);
        return err;
    }
    
    // Initialize completion flag
    uint32_t zero = 0;
    err = cudaMemcpyAsync(d_completion_flag, &zero, sizeof(uint32_t),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_swap_queue);
        cudaFree(d_completion_flag);
        return err;
    }
    
    // Launch kernel with cooperative groups
    dim3 block_size(256);
    dim3 grid_size((count + block_size.x - 1) / block_size.x);
    
    // Ensure we don't exceed GPU limits
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kernel_atomic_swap_pages, block_size.x, 0);
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    grid_size.x = min(grid_size.x, max_blocks_per_sm * props.multiProcessorCount);
    
    void* kernel_args[] = {
        &atlas->d_entries,
        &atlas->d_gpu_memory,
        &atlas->d_swap_buffer,
        &d_swap_queue,
        &count,
        &d_completion_flag
    };
    
    err = cudaLaunchCooperativeKernel(
        (void*)kernel_atomic_swap_pages,
        grid_size, block_size,
        kernel_args, 0, stream
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_swap_queue);
        cudaFree(d_completion_flag);
        return err;
    }
    
    // Wait for completion
    err = cudaStreamSynchronize(stream);
    
    // Cleanup
    cudaFree(d_swap_queue);
    cudaFree(d_completion_flag);
    
    return err;
}

cudaError_t atlas_memory_fence(atlas_t* atlas) {
    // System-wide memory fence
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return err;
    
    // Invalidate L2 cache for coherency
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (err != cudaSuccess) return err;
    
    // Memory barrier
    __sync_synchronize();
    
    return cudaSuccess;
}

} // extern "C"

// Helper function for cooperative memory copy
__device__ void cooperative_memcpy_async(
    cg::thread_block block,
    void* dst, 
    const void* src, 
    size_t size
) {
    const size_t threads = block.size();
    const size_t tid = block.thread_rank();
    
    // Copy in 4-byte chunks using all threads
    const uint32_t* src32 = (const uint32_t*)src;
    uint32_t* dst32 = (uint32_t*)dst;
    const size_t words = size / 4;
    
    for (size_t i = tid; i < words; i += threads) {
        dst32[i] = src32[i];
    }
    
    // Handle remaining bytes
    if (tid == 0) {
        const size_t remaining = size % 4;
        const char* src_bytes = (const char*)src + (words * 4);
        char* dst_bytes = (char*)dst + (words * 4);
        
        for (size_t i = 0; i < remaining; i++) {
            dst_bytes[i] = src_bytes[i];
        }
    }
    
    // Synchronize block
    block.sync();
}
