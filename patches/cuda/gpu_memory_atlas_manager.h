/*
 * GGUF Shard Atlas Header
 * 
 * Defines data structures and API for CUDA-based sharded GGUF memory management
 */

#ifndef GGUF_SHARD_ATLAS_H
#define GGUF_SHARD_ATLAS_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Atlas entry structure (matches spec)
typedef struct {
    uint64_t virtual_addr;    // Virtual address in model space
    uint64_t physical_addr;   // Physical GPU memory address  
    uint32_t shard_id;        // Shard identifier
    uint32_t page_offset;     // Offset within shard
    uint8_t  state;           // RESIDENT, SWAPPED, PENDING, LOCKED
    uint8_t  priority;        // Access priority (0-255)
    uint16_t ref_count;       // Reference counter
} atlas_entry_t;

// Atlas structure
typedef struct {
    atlas_entry_t* entries;      // Host atlas entries
    atlas_entry_t* d_entries;    // Device atlas entries
    uint32_t entry_count;        // Number of entries
    uint32_t capacity;           // Maximum entries
    
    void* d_gpu_memory;          // GPU memory pool
    void* d_swap_buffer;         // Swap buffer
    size_t memory_size;          // Total memory size
    size_t swap_size;            // Swap buffer size
    
    cudaStream_t stream;         // CUDA stream for operations
} atlas_t;

// API Functions

/**
 * Initialize atlas with given capacity
 */
cudaError_t atlas_init(atlas_t* atlas, uint32_t capacity, size_t memory_size);

/**
 * Cleanup atlas resources
 */
void atlas_cleanup(atlas_t* atlas);

/**
 * Lookup atlas entry by virtual address
 */
atlas_entry_t* atlas_lookup(atlas_t* atlas, uint64_t virtual_addr);

/**
 * Perform atomic swap of multiple pages
 */
cudaError_t atlas_atomic_swap(
    atlas_t* atlas,
    uint32_t* shard_ids,
    uint32_t count,
    cudaStream_t stream
);

/**
 * Execute memory fence for coherency
 */
cudaError_t atlas_memory_fence(atlas_t* atlas);

/**
 * Add entry to atlas
 */
cudaError_t atlas_add_entry(
    atlas_t* atlas,
    uint64_t virtual_addr,
    uint64_t physical_addr,
    uint32_t shard_id,
    uint8_t priority
);

/**
 * Remove entry from atlas
 */
cudaError_t atlas_remove_entry(atlas_t* atlas, uint32_t shard_id);

/**
 * Update entry state atomically
 */
cudaError_t atlas_update_state(
    atlas_t* atlas,
    uint32_t shard_id,
    uint8_t new_state
);

/**
 * Get memory usage statistics
 */
typedef struct {
    uint32_t resident_pages;
    uint32_t swapped_pages;
    uint32_t pending_pages;
    size_t memory_used;
    size_t swap_used;
    double hit_ratio;
} atlas_stats_t;

cudaError_t atlas_get_stats(atlas_t* atlas, atlas_stats_t* stats);

// CUDA kernel declarations
__global__ void kernel_atomic_swap_pages(
    atlas_entry_t* atlas,
    void* gpu_memory,
    void* swap_buffer,
    uint32_t* swap_queue,
    uint32_t queue_size,
    volatile uint32_t* completion_flag
);

__device__ void cooperative_memcpy_async(
    cooperative_groups::thread_block block,
    void* dst, 
    const void* src, 
    size_t size
);

#ifdef __cplusplus
}
#endif

#endif // GGUF_SHARD_ATLAS_H
