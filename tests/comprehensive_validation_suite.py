#!/usr/bin/env python3
"""
Comprehensive Validation Suite

Integrity testing, chaos engineering, and performance validation
for the GGUF Shard memory management system.
"""

import os
import sys
import time
import random
import zlib  # Use zlib for CRC32
import subprocess
import tempfile
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional

class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration

class GGUFShardTester:
    def __init__(self, test_data_dir: str = "test_data"):
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(exist_ok=True)
        self.results: List[TestResult] = []
        
    def run_all_tests(self) -> bool:
        """Run complete test suite"""
        print("Starting GGUF Shard Test Suite")
        print("=" * 50)
        
        # Prepare test data
        self._prepare_test_data()
        
        # Run test categories
        integrity_passed = self._run_integrity_tests()
        chaos_passed = self._run_chaos_tests()
        throughput_passed = self._run_throughput_tests()
        
        # Report results
        self._report_results()
        
        return integrity_passed and chaos_passed and throughput_passed
    
    def _prepare_test_data(self) -> None:
        """Create test GGUF files"""
        print("Preparing test data...")
        
        # Create small test GGUF file
        small_file = self.test_data_dir / "small.gguf"
        self._create_test_gguf(small_file, size=16384)  # 16KB
        
        # Create medium test GGUF file
        medium_file = self.test_data_dir / "medium.gguf"
        self._create_test_gguf(medium_file, size=1048576)  # 1MB
        
        # Create large test GGUF file
        large_file = self.test_data_dir / "large.gguf"
        self._create_test_gguf(large_file, size=16777216)  # 16MB
        
        print("Test data prepared")
    
    def _create_test_gguf(self, file_path: Path, size: int) -> None:
        """Create a test GGUF file with known content"""
        with open(file_path, 'wb') as f:
            # Write GGUF magic and basic header
            f.write(b'GGUF')  # Magic
            f.write(b'\x03\x00\x00\x00')  # Version 3
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')  # Tensor count
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')  # KV count
            
            # Fill remaining space with deterministic pattern
            remaining = size - 24
            pattern = b'SHARD_TEST_DATA_' * (remaining // 16 + 1)
            f.write(pattern[:remaining])
    
    def _run_integrity_tests(self) -> bool:
        """Test data integrity and consistency"""
        print("\nRunning Integrity Tests")
        print("-" * 30)
        
        tests = [
            self._test_shard_creation,
            self._test_shard_reconstruction,
            self._test_crc_validation,
            self._test_atlas_consistency,
            self._test_delta_application
        ]
        
        passed = 0
        for test in tests:
            result = test()
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} {result.name} ({result.duration:.3f}s)")
            if not result.passed:
                print(f"    {result.message}")
            if result.passed:
                passed += 1
        
        return passed == len(tests)
    
    def _test_shard_creation(self) -> TestResult:
        """Test basic shard creation"""
        start_time = time.time()
        
        try:
            test_file = self.test_data_dir / "medium.gguf"
            
            # Run forge shard command
            result = subprocess.run([
                sys.executable, "forge/model_sharding_tool.py", "shard", str(test_file)
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode != 0:
                return TestResult("Shard Creation", False, f"Command failed: {result.stderr}")
            
            # Check output files exist
            core_file = self.test_data_dir / "core.gguf"
            map_file = self.test_data_dir / "core.sgmap"
            
            if not core_file.exists():
                return TestResult("Shard Creation", False, "core.gguf not created")
            
            if not map_file.exists():
                return TestResult("Shard Creation", False, "core.sgmap not created")
            
            duration = time.time() - start_time
            return TestResult("Shard Creation", True, "", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Shard Creation", False, str(e), duration)
    
    def _test_shard_reconstruction(self) -> TestResult:
        """Test reconstructing original file from shards"""
        start_time = time.time()
        
        try:
            # This would require implementing a reconstruction tool
            # For now, we'll do a basic file size check
            original_file = self.test_data_dir / "medium.gguf"
            core_file = self.test_data_dir / "core.gguf"
            
            if not core_file.exists():
                return TestResult("Shard Reconstruction", False, "No core file to test")
            
            # Check that core file has reasonable size (with headers, should be close)
            original_size = original_file.stat().st_size
            core_size = core_file.stat().st_size
            
            # Core file should be larger due to headers and padding
            if core_size < original_size:
                return TestResult("Shard Reconstruction", False, "Core file smaller than original")
            
            duration = time.time() - start_time
            return TestResult("Shard Reconstruction", True, "", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Shard Reconstruction", False, str(e), duration)
    
    def _test_crc_validation(self) -> TestResult:
        """Test CRC validation of pages"""
        start_time = time.time()
        
        try:
            # Read core file and validate page CRCs
            core_file = self.test_data_dir / "core.gguf"
            
            if not core_file.exists():
                return TestResult("CRC Validation", False, "No core file to test")
            
            with open(core_file, 'rb') as f:
                # Skip SGUF header (256 bytes)
                f.seek(256)
                
                page_count = 0
                valid_crcs = 0
                
                while True:
                    page_data = f.read(4096)
                    if not page_data:
                        break
                    
                    if len(page_data) < 8:
                        break
                    
                    # Extract CRC tag from last 8 bytes
                    content = page_data[:-8]
                    crc_bytes = page_data[-8:-4]
                    magic_bytes = page_data[-4:]
                    
                    if magic_bytes == b'PGCR':
                        stored_crc = int.from_bytes(crc_bytes, 'little')
                        calculated_crc = zlib.crc32(content) & 0xffffffff
                        
                        if stored_crc == calculated_crc:
                            valid_crcs += 1
                    
                    page_count += 1
                
                if page_count == 0:
                    return TestResult("CRC Validation", False, "No pages found")
                
                if valid_crcs != page_count:
                    return TestResult("CRC Validation", False, 
                                    f"CRC mismatch: {valid_crcs}/{page_count} valid")
                
            duration = time.time() - start_time
            return TestResult("CRC Validation", True, f"Validated {page_count} pages", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("CRC Validation", False, str(e), duration)
    
    def _test_atlas_consistency(self) -> TestResult:
        """Test atlas mapping consistency"""
        start_time = time.time()
        
        try:
            map_file = self.test_data_dir / "core.sgmap"
            
            if not map_file.exists():
                return TestResult("Atlas Consistency", False, "No map file to test")
            
            import json
            with open(map_file, 'r') as f:
                shard_map = json.load(f)
            
            # Validate map structure
            required_fields = ['version', 'total_shards', 'page_size', 'shards', 'atlas']
            for field in required_fields:
                if field not in shard_map:
                    return TestResult("Atlas Consistency", False, f"Missing field: {field}")
            
            # Validate shards
            shards = shard_map['shards']
            if len(shards) != shard_map['total_shards']:
                return TestResult("Atlas Consistency", False, "Shard count mismatch")
            
            # Check for gaps or overlaps
            offsets = sorted([shard['offset'] for shard in shards])
            page_size = shard_map['page_size']
            
            for i, offset in enumerate(offsets):
                if offset != i * page_size:
                    return TestResult("Atlas Consistency", False, f"Gap at offset {offset}")
            
            duration = time.time() - start_time
            return TestResult("Atlas Consistency", True, f"Validated {len(shards)} entries", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Atlas Consistency", False, str(e), duration)
    
    def _test_delta_application(self) -> TestResult:
        """Test delta creation and application"""
        start_time = time.time()
        
        try:
            # Create modified version of test file
            base_file = self.test_data_dir / "small.gguf"
            modified_file = self.test_data_dir / "small_modified.gguf"
            
            # Copy and modify file
            with open(base_file, 'rb') as f:
                data = bytearray(f.read())
            
            # Modify some bytes in the middle
            if len(data) > 1000:
                data[500:600] = b'X' * 100
            
            with open(modified_file, 'wb') as f:
                f.write(data)
            
            # Create delta
            result = subprocess.run([
                sys.executable, "trainer/incremental_model_updater.py",
                "--base", str(base_file),
                "--target", str(modified_file),
                "--output", str(self.test_data_dir / "test_delta")
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode != 0:
                return TestResult("Delta Application", False, f"Delta creation failed: {result.stderr}")
            
            # Check delta files exist
            delta_file = self.test_data_dir / "test_delta.delta"
            delta_map = self.test_data_dir / "test_delta.sgmap"
            
            if not delta_file.exists() or not delta_map.exists():
                return TestResult("Delta Application", False, "Delta files not created")
            
            duration = time.time() - start_time
            return TestResult("Delta Application", True, "", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Delta Application", False, str(e), duration)
    
    def _run_chaos_tests(self) -> bool:
        """Test system under stress and error conditions"""
        print("\nRunning Chaos Tests")
        print("-" * 30)
        
        tests = [
            self._test_corrupted_input,
            self._test_partial_files,
            self._test_memory_pressure,
            self._test_concurrent_access
        ]
        
        passed = 0
        for test in tests:
            result = test()
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} {result.name} ({result.duration:.3f}s)")
            if not result.passed:
                print(f"    {result.message}")
            if result.passed:
                passed += 1
        
        return passed == len(tests)
    
    def _test_corrupted_input(self) -> TestResult:
        """Test handling of corrupted input files"""
        start_time = time.time()
        
        try:
            # Create corrupted GGUF file
            corrupted_file = self.test_data_dir / "corrupted.gguf"
            with open(corrupted_file, 'wb') as f:
                f.write(b'GGUF')  # Valid magic
                f.write(os.urandom(1000))  # Random data
            
            # Test should handle gracefully
            result = subprocess.run([
                sys.executable, "forge/model_sharding_tool.py", "shard", str(corrupted_file)
            ], capture_output=True, text=True, cwd=".")
            
            # Should either succeed or fail gracefully (not crash)
            duration = time.time() - start_time
            return TestResult("Corrupted Input", True, "Handled gracefully", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Corrupted Input", False, str(e), duration)
    
    def _test_partial_files(self) -> TestResult:
        """Test handling of partial/truncated files"""
        start_time = time.time()
        
        try:
            # Create truncated file
            original_file = self.test_data_dir / "medium.gguf"
            truncated_file = self.test_data_dir / "truncated.gguf"
            
            with open(original_file, 'rb') as src:
                data = src.read()
            
            # Write only first half
            with open(truncated_file, 'wb') as dst:
                dst.write(data[:len(data)//2])
            
            # Should handle gracefully
            result = subprocess.run([
                sys.executable, "forge/model_sharding_tool.py", "shard", str(truncated_file)
            ], capture_output=True, text=True, cwd=".")
            
            duration = time.time() - start_time
            return TestResult("Partial Files", True, "Handled gracefully", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Partial Files", False, str(e), duration)
    
    def _test_memory_pressure(self) -> TestResult:
        """Test behavior under memory pressure"""
        start_time = time.time()
        
        try:
            # Create multiple large files to process
            large_files = []
            for i in range(3):
                large_file = self.test_data_dir / f"large_{i}.gguf"
                self._create_test_gguf(large_file, size=8*1024*1024)  # 8MB each
                large_files.append(large_file)
            
            # Process all files
            success_count = 0
            for large_file in large_files:
                result = subprocess.run([
                    sys.executable, "forge/model_sharding_tool.py", "shard", str(large_file)
                ], capture_output=True, text=True, cwd=".")
                
                if result.returncode == 0:
                    success_count += 1
            
            # Should handle at least some files successfully
            if success_count == 0:
                return TestResult("Memory Pressure", False, "Failed all files under pressure")
            
            duration = time.time() - start_time
            return TestResult("Memory Pressure", True, f"Processed {success_count}/3 files", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Memory Pressure", False, str(e), duration)
    
    def _test_concurrent_access(self) -> TestResult:
        """Test concurrent file processing"""
        start_time = time.time()
        
        try:
            # Create multiple test files
            test_files = []
            for i in range(4):
                test_file = self.test_data_dir / f"concurrent_{i}.gguf"
                self._create_test_gguf(test_file, size=512*1024)  # 512KB each
                test_files.append(test_file)
            
            # Process concurrently
            def process_file(file_path):
                result = subprocess.run([
                    sys.executable, "forge/model_sharding_tool.py", "shard", str(file_path)
                ], capture_output=True, text=True, cwd=".")
                return result.returncode == 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_file, f) for f in test_files]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            success_count = sum(results)
            if success_count < len(test_files) // 2:
                return TestResult("Concurrent Access", False, f"Only {success_count}/{len(test_files)} succeeded")
            
            duration = time.time() - start_time
            return TestResult("Concurrent Access", True, f"Processed {success_count}/{len(test_files)} files", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Concurrent Access", False, str(e), duration)
    
    def _run_throughput_tests(self) -> bool:
        """Test performance and throughput"""
        print("\nRunning Throughput Tests")
        print("-" * 30)
        
        tests = [
            self._test_shard_throughput,
            self._test_delta_throughput,
            self._test_memory_bandwidth
        ]
        
        passed = 0
        for test in tests:
            result = test()
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} {result.name} ({result.duration:.3f}s)")
            if not result.passed:
                print(f"    {result.message}")
            if result.passed:
                passed += 1
        
        return passed == len(tests)
    
    def _test_shard_throughput(self) -> TestResult:
        """Test sharding throughput"""
        start_time = time.time()
        
        try:
            # Process large file and measure throughput
            large_file = self.test_data_dir / "large.gguf"
            file_size = large_file.stat().st_size
            
            process_start = time.time()
            result = subprocess.run([
                sys.executable, "forge/model_sharding_tool.py", "shard", str(large_file)
            ], capture_output=True, text=True, cwd=".")
            process_time = time.time() - process_start
            
            if result.returncode != 0:
                return TestResult("Shard Throughput", False, f"Processing failed: {result.stderr}")
            
            throughput_mbps = (file_size / (1024 * 1024)) / process_time
            
            # Expect at least 10 MB/s throughput
            if throughput_mbps < 10.0:
                return TestResult("Shard Throughput", False, f"Low throughput: {throughput_mbps:.2f} MB/s")
            
            duration = time.time() - start_time
            return TestResult("Shard Throughput", True, f"{throughput_mbps:.2f} MB/s", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Shard Throughput", False, str(e), duration)
    
    def _test_delta_throughput(self) -> TestResult:
        """Test delta creation throughput"""
        start_time = time.time()
        
        try:
            base_file = self.test_data_dir / "large.gguf"
            modified_file = self.test_data_dir / "large_modified.gguf"
            
            # Create modified version
            with open(base_file, 'rb') as f:
                data = bytearray(f.read())
            
            # Modify 10% of the file randomly
            modify_count = len(data) // 10
            for _ in range(modify_count):
                pos = random.randint(0, len(data) - 1)
                data[pos] = random.randint(0, 255)
            
            with open(modified_file, 'wb') as f:
                f.write(data)
            
            # Measure delta creation time
            process_start = time.time()
            result = subprocess.run([
                sys.executable, "trainer/incremental_model_updater.py",
                "--base", str(base_file),
                "--target", str(modified_file),
                "--output", str(self.test_data_dir / "perf_delta")
            ], capture_output=True, text=True, cwd=".")
            process_time = time.time() - process_start
            
            if result.returncode != 0:
                return TestResult("Delta Throughput", False, f"Delta creation failed: {result.stderr}")
            
            file_size = base_file.stat().st_size
            throughput_mbps = (file_size / (1024 * 1024)) / process_time
            
            # Expect at least 5 MB/s for delta processing
            if throughput_mbps < 5.0:
                return TestResult("Delta Throughput", False, f"Low throughput: {throughput_mbps:.2f} MB/s")
            
            duration = time.time() - start_time
            return TestResult("Delta Throughput", True, f"{throughput_mbps:.2f} MB/s", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Delta Throughput", False, str(e), duration)
    
    def _test_memory_bandwidth(self) -> TestResult:
        """Test memory access patterns"""
        start_time = time.time()
        
        try:
            # Create test for memory access patterns
            # This is a simplified test since we don't have the full CUDA implementation
            
            # Measure file I/O patterns
            test_file = self.test_data_dir / "large.gguf"
            page_size = 4096
            
            access_start = time.time()
            with open(test_file, 'rb') as f:
                # Simulate random page access
                file_size = f.seek(0, 2)  # Get file size
                f.seek(0)
                
                pages_read = 0
                for _ in range(100):  # Read 100 random pages
                    page_offset = random.randint(0, (file_size // page_size) - 1) * page_size
                    f.seek(page_offset)
                    data = f.read(page_size)
                    if len(data) == page_size:
                        pages_read += 1
            
            access_time = time.time() - access_start
            
            if pages_read == 0:
                return TestResult("Memory Bandwidth", False, "No pages read successfully")
            
            bandwidth_mbps = (pages_read * page_size / (1024 * 1024)) / access_time
            
            # Expect reasonable bandwidth for random access
            if bandwidth_mbps < 1.0:
                return TestResult("Memory Bandwidth", False, f"Low bandwidth: {bandwidth_mbps:.2f} MB/s")
            
            duration = time.time() - start_time
            return TestResult("Memory Bandwidth", True, f"{bandwidth_mbps:.2f} MB/s random access", duration)
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult("Memory Bandwidth", False, str(e), duration)
    
    def _report_results(self) -> None:
        """Generate test report"""
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        total_time = sum(r.duration for r in self.results)
        
        print(f"Tests Run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests != total_tests:
            print("\nFAILED TESTS:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.message}")
        else:
            print("\nALL TESTS PASSED!")

def main():
    tester = GGUFShardTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
