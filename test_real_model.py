#!/usr/bin/env python3
"""
Real Model Test Suite

Tests the GGUF Shard Suite with a real model file to ensure
end-to-end functionality works correctly.
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

def test_real_model_workflow(model_file: str):
    """Test complete workflow with real model file"""
    print(f"🧪 Testing GGUF Shard Suite with: {model_file}")
    print("=" * 60)
    
    model_path = Path(model_file)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_file}")
        return False
        
    file_size = model_path.stat().st_size
    print(f"📁 Model size: {file_size / (1024*1024):.2f} MB")
    
    # Test 1: Shard the model
    print("\n🔨 Step 1: Creating shards...")
    start_time = time.time()
    
    result = subprocess.run([
        sys.executable, "forge/model_sharding_tool.py", "shard", str(model_path)
    ], capture_output=True, text=True)
    
    shard_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ Sharding failed: {result.stderr}")
        return False
        
    print(f"✅ Sharding completed in {shard_time:.2f}s")
    
    # Check output files
    stem = model_path.stem  # Keep stem for other file names
    core_file = Path("core.gguf")
    map_file = Path("core.sgmap")
    
    if not core_file.exists():
        print(f"❌ Core file not created: {core_file}")
        return False
        
    if not map_file.exists():
        print(f"❌ Map file not created: {map_file}")
        return False
        
    core_size = core_file.stat().st_size
    print(f"📁 Core file size: {core_size / (1024*1024):.2f} MB")
    
    # Test 2: Validate shard map
    print("\n🗺️ Step 2: Validating shard map...")
    try:
        with open(map_file, 'r') as f:
            shard_map = json.load(f)
            
        required_fields = ['version', 'total_shards', 'page_size', 'shards', 'atlas']
        for field in required_fields:
            if field not in shard_map:
                print(f"❌ Missing field in map: {field}")
                return False
                
        print(f"✅ Map validation passed")
        print(f"   - Version: {shard_map.get('version', 'unknown')}")
        print(f"   - Total shards: {shard_map.get('total_shards', 0)}")
        print(f"   - Page size: {shard_map.get('page_size', 0)} bytes")
        
    except Exception as e:
        print(f"❌ Map validation failed: {e}")
        return False
    
    # Test 3: Create a delta (simulate a small change)
    print("\n🔄 Step 3: Testing delta creation...")
    
    # Create a slightly modified version
    modified_file = Path(f"{stem}_modified.gguf")
    try:
        with open(model_path, 'rb') as src:
            data = bytearray(src.read())
        
        # Modify just a few bytes (simulate minimal change)
        if len(data) > 1000:
            data[500:510] = b'X' * 10
            
        with open(modified_file, 'wb') as dst:
            dst.write(data)
            
        # Create delta
        delta_start = time.time()
        result = subprocess.run([
            sys.executable, "trainer/incremental_model_updater.py",
            "--base", str(model_path),
            "--target", str(modified_file),
            "--output", f"{stem}_test_delta"
        ], capture_output=True, text=True)
        
        delta_time = time.time() - delta_start
        
        if result.returncode != 0:
            print(f"❌ Delta creation failed: {result.stderr}")
            return False
            
        print(f"✅ Delta creation completed in {delta_time:.2f}s")
        
        # Check delta files
        delta_file = Path(f"{stem}_test_delta.delta")
        delta_map = Path(f"{stem}_test_delta.sgmap")
        
        if delta_file.exists():
            delta_size = delta_file.stat().st_size
            compression_ratio = (delta_size / file_size) * 100
            print(f"📁 Delta size: {delta_size / 1024:.2f} KB ({compression_ratio:.3f}% of original)")
        
    except Exception as e:
        print(f"❌ Delta test failed: {e}")
        return False
    
    # Test 4: Performance metrics
    print("\n📊 Step 4: Performance Summary")
    print("-" * 40)
    throughput_mb = (file_size / (1024*1024)) / shard_time
    print(f"Sharding throughput: {throughput_mb:.2f} MB/s")
    
    if core_size > 0:
        efficiency = ((file_size - core_size) / file_size) * 100
        print(f"Storage efficiency: {efficiency:.2f}% space saved")
    
    return True

def cleanup_test_files(model_stem: str):
    """Clean up all test-generated files"""
    print("\n🧹 Cleaning up test files...")
    
    cleanup_patterns = [
        f"{model_stem}_core.gguf",
        f"{model_stem}_core.sgmap", 
        f"{model_stem}_modified.gguf",
        f"{model_stem}_test_delta.delta",
        f"{model_stem}_test_delta.sgmap"
    ]
    
    cleaned = 0
    for pattern in cleanup_patterns:
        file_path = Path(pattern)
        if file_path.exists():
            file_path.unlink()
            cleaned += 1
            print(f"   Removed: {pattern}")
    
    print(f"✅ Cleaned up {cleaned} test files")

def main():
    # Find GGUF files in current directory (exclude processed files)
    all_gguf_files = list(Path('.').glob('*.gguf'))
    # Filter out generated files that start with 'core'
    gguf_files = [f for f in all_gguf_files if not f.name.startswith('core')]
    
    if not gguf_files:
        print("❌ No original GGUF files found in current directory")
        print("Available files:", [f.name for f in all_gguf_files])
        return False
    
    print(f"Found {len(gguf_files)} GGUF file(s):")
    for f in gguf_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    success = True
    for model_file in gguf_files:
        try:
            if not test_real_model_workflow(str(model_file)):
                success = False
                break
                
            print(f"\n✅ All tests passed for {model_file.name}!")
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            success = False
            break
        finally:
            # Always cleanup test files
            cleanup_test_files(model_file.stem)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 REAL MODEL WORKFLOW TEST: PASSED")
        print("The GGUF Shard Suite is working correctly!")
    else:
        print("❌ REAL MODEL WORKFLOW TEST: FAILED") 
        print("Issues detected in the GGUF Shard Suite")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
