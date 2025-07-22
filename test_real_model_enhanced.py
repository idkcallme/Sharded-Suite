#!/usr/bin/env python3
"""
Enhanced Real Model Test Suite

Tests the GGUF Shard Suite with a real model file to ensure
end-to-end functionality works correctly with performance monitoring.
"""

import os
import sys
import time
import subprocess
import json
import logging
import threading
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Optional
import psutil

@dataclass
class PerformanceMetrics:
    """Performance metrics collection"""
    cpu_usage: List[float]
    memory_usage: List[float]
    duration: float
    throughput_mb_s: float
    peak_memory_mb: float

class PerformanceMonitor:
    """Monitor system performance during operations"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = PerformanceMetrics([], [], 0, 0, 0)
        
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        
        self.metrics.duration = time.time() - self.start_time
        if self.metrics.memory_usage:
            self.metrics.peak_memory_mb = max(self.metrics.memory_usage)
        
        return self.metrics
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self.metrics.cpu_usage.append(psutil.cpu_percent())
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                self.metrics.memory_usage.append(memory_mb)
                time.sleep(0.1)
            except Exception as e:
                logging.warning(f"Performance monitoring error: {e}")

@contextmanager
def test_step(step_name: str):
    """Context manager for test steps with proper error handling"""
    print(f"\n[STEP] {step_name}...")
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        print(f"[PASS] {step_name} completed in {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start_time
        print(f"[FAIL] {step_name} failed after {duration:.2f}s: {e}")
        logging.error(f"{step_name} failed: {e}", exc_info=True)
        raise

def setup_logging():
    """Setup detailed logging for test execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_execution.log'),
            logging.StreamHandler()
        ]
    )

class RegressionTester:
    """Track performance over time and detect regressions"""
    
    def __init__(self):
        self.baseline_file = Path("test_baselines.json")
        self.load_baselines()
        
    def load_baselines(self):
        """Load performance baselines from file"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    self.baselines = json.load(f)
            except json.JSONDecodeError:
                self.baselines = {}
        else:
            self.baselines = {}
            
    def save_baselines(self):
        """Save current baselines to file"""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)
            
    def check_regression(self, test_name: str, current_value: float, threshold: float = 0.15):
        """Check if current performance is a regression"""
        if test_name not in self.baselines:
            self.baselines[test_name] = {
                "baseline": current_value,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "runs": 1
            }
            self.save_baselines()
            return False, f"New baseline established: {current_value:.2f}"
            
        baseline = self.baselines[test_name]["baseline"]
        regression_threshold = baseline * (1 + threshold)
        
        if current_value > regression_threshold:
            return True, f"REGRESSION DETECTED: {current_value:.2f} > {regression_threshold:.2f} (threshold: +{threshold*100:.0f}%)"
        
        # Update running average
        runs = self.baselines[test_name]["runs"]
        new_baseline = (baseline * runs + current_value) / (runs + 1)
        self.baselines[test_name]["baseline"] = new_baseline
        self.baselines[test_name]["runs"] = runs + 1
        self.baselines[test_name]["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.save_baselines()
        
        return False, f"Performance within threshold: {current_value:.2f} (baseline: {baseline:.2f})"

def test_real_model_workflow(model_file: str, regression_tester: RegressionTester = None):
    """Test complete workflow with real model file"""
    print(f"Testing GGUF Shard Suite with: {model_file}")
    print("=" * 80)
    
    model_path = Path(model_file)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_file}")
        return False
        
    file_size = model_path.stat().st_size
    print(f"Model size: {file_size / (1024*1024):.2f} MB")
    
    # Initialize performance monitoring
    monitor = PerformanceMonitor()
    
    # Test 1: Shard the model
    with test_step("Creating shards"):
        monitor.start_monitoring()
        
        result = subprocess.run([
            sys.executable, "forge/model_sharding_tool.py", "shard", str(model_path)
        ], capture_output=True, text=True)
        
        metrics = monitor.stop_monitoring()
        
        if result.returncode != 0:
            print(f"ERROR: Sharding failed: {result.stderr}")
            return False
            
        # Check output files
        stem = model_path.stem
        core_file = Path("core.gguf")
        map_file = Path("core.sgmap")
        
        if not core_file.exists():
            print(f"ERROR: Core file not created: {core_file}")
            return False
            
        if not map_file.exists():
            print(f"ERROR: Map file not created: {map_file}")
            return False
        
        core_size = core_file.stat().st_size
        print(f"Core file size: {core_size / (1024*1024):.2f} MB")
        
        if metrics.cpu_usage and metrics.memory_usage:
            avg_cpu = sum(metrics.cpu_usage) / len(metrics.cpu_usage)
            print(f"Performance: CPU avg {avg_cpu:.1f}%, Peak memory {metrics.peak_memory_mb:.0f} MB")
            
            # Check for regression
            if regression_tester:
                throughput = (file_size / (1024*1024)) / metrics.duration
                is_regression, message = regression_tester.check_regression(
                    f"sharding_throughput_mb_s", throughput
                )
                print(f"Throughput: {message}")
                if is_regression:
                    print("WARNING: Performance regression detected!")

    # Test 2: Validate shard map
    with test_step("Validating shard map"):
        with open(map_file, 'r') as f:
            shard_map = json.load(f)
            
        required_fields = ['version', 'total_shards', 'page_size', 'shards', 'atlas']
        for field in required_fields:
            if field not in shard_map:
                print(f"ERROR: Missing field in map: {field}")
                return False
                
        print(f"Map validation passed")
        print(f"   - Version: {shard_map.get('version', 'unknown')}")
        print(f"   - Total shards: {shard_map.get('total_shards', 0)}")
        print(f"   - Page size: {shard_map.get('page_size', 0)} bytes")
    
    # Test 3: Create a delta (simulate a small change)
    with test_step("Testing delta creation"):
        modified_file = Path(f"{stem}_modified.gguf")
        
        with open(model_path, 'rb') as src:
            data = bytearray(src.read())
        
        # Modify just a few bytes (simulate minimal change)
        if len(data) > 1000:
            data[500:510] = b'X' * 10
            
        with open(modified_file, 'wb') as dst:
            dst.write(data)
            
        # Create delta
        delta_monitor = PerformanceMonitor()
        delta_monitor.start_monitoring()
        
        result = subprocess.run([
            sys.executable, "trainer/incremental_model_updater.py",
            "--base", str(model_path),
            "--target", str(modified_file),
            "--output", f"{stem}_test_delta"
        ], capture_output=True, text=True)
        
        delta_metrics = delta_monitor.stop_monitoring()
        
        if result.returncode != 0:
            print(f"ERROR: Delta creation failed: {result.stderr}")
            return False
            
        print(f"Delta creation completed in {delta_metrics.duration:.2f}s")
        
        # Check delta files
        delta_file = Path(f"{stem}_test_delta.delta")
        delta_map_file = Path(f"{stem}_test_delta.sgmap")
        
        if delta_file.exists():
            delta_size = delta_file.stat().st_size
            compression_ratio = (delta_size / file_size) * 100
            print(f"Delta size: {delta_size / 1024:.2f} KB ({compression_ratio:.3f}% of original)")
            
            # Check delta performance regression
            if regression_tester:
                delta_throughput = (file_size / (1024*1024)) / delta_metrics.duration
                is_regression, message = regression_tester.check_regression(
                    f"delta_throughput_mb_s", delta_throughput
                )
                print(f"Delta throughput: {message}")

    # Test 4: Memory stress test (if file is large enough)
    if file_size > 100 * 1024 * 1024:  # 100MB+
        with test_step("Memory stress test"):
            # Monitor memory usage during multiple operations
            stress_monitor = PerformanceMonitor()
            stress_monitor.start_monitoring()
            
            # Simulate concurrent access
            for i in range(3):
                with open(core_file, 'rb') as f:
                    chunk = f.read(1024 * 1024)  # Read 1MB chunks
                    
            stress_metrics = stress_monitor.stop_monitoring()
            
            if stress_metrics.memory_usage:
                max_memory = max(stress_metrics.memory_usage)
                print(f"Peak memory during stress test: {max_memory:.0f} MB")

    # Test 5: File integrity check
    with test_step("File integrity verification"):
        # Verify core file is valid GGUF
        with open(core_file, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'GGUF'):
                print("ERROR: Core file does not have valid GGUF header")
                return False
        print("Core file integrity verified")

    # Performance summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    throughput_mb = (file_size / (1024*1024)) / metrics.duration
    print(f"Sharding throughput: {throughput_mb:.2f} MB/s")
    
    if core_size > 0:
        efficiency = ((file_size - core_size) / file_size) * 100
        print(f"Storage efficiency: {efficiency:.2f}% space saved")
    
    return True

def cleanup_test_files(model_stem: str):
    """Clean up all test-generated files with enhanced patterns"""
    print("\nCleaning up test files...")
    
    cleanup_patterns = [
        "core.gguf",
        "core.sgmap",
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
    
    print(f"Cleaned up {cleaned} test files")

def run_comprehensive_test_suite():
    """Run the complete test suite with all models"""
    setup_logging()
    regression_tester = RegressionTester()
    
    # Find GGUF files in current directory (exclude processed files)
    all_gguf_files = list(Path('.').glob('*.gguf'))
    gguf_files = [f for f in all_gguf_files if not f.name.startswith('core')]
    
    if not gguf_files:
        print("ERROR: No original GGUF files found in current directory")
        if all_gguf_files:
            print("Available files:", [f.name for f in all_gguf_files])
        return False
    
    print("GGUF Shard Suite - Enhanced Test Runner")
    print("=" * 80)
    print(f"Found {len(gguf_files)} GGUF file(s):")
    for f in gguf_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    success = True
    total_start = time.time()
    
    for model_file in gguf_files:
        try:
            logging.info(f"Starting test for {model_file.name}")
            
            if not test_real_model_workflow(str(model_file), regression_tester):
                success = False
                break
                
            print(f"\n[PASS] All tests passed for {model_file.name}")
            logging.info(f"Test completed successfully for {model_file.name}")
            
        except Exception as e:
            print(f"[FAIL] Test failed with error: {e}")
            logging.error(f"Test failed for {model_file.name}: {e}", exc_info=True)
            success = False
            break
        finally:
            cleanup_test_files(model_file.stem)
    
    total_duration = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Total execution time: {total_duration:.2f}s")
    
    if success:
        print("STATUS: PASSED - The GGUF Shard Suite is working correctly!")
        logging.info("All tests passed successfully")
        return True
    else:
        print("STATUS: FAILED - Issues detected in the GGUF Shard Suite")
        logging.error("Test suite failed")
        return False

def main():
    """Main entry point with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced GGUF Shard Suite Test Runner")
    parser.add_argument("--model", help="Specific model file to test")
    parser.add_argument("--no-regression", action="store_true", help="Skip regression testing")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Set logging level")
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_execution.log'),
            logging.StreamHandler()
        ]
    )
    
    if args.model:
        # Test specific model
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: Model file not found: {args.model}")
            return False
            
        regression_tester = None if args.no_regression else RegressionTester()
        success = test_real_model_workflow(args.model, regression_tester)
        cleanup_test_files(model_path.stem)
        return success
    else:
        # Run comprehensive test suite
        return run_comprehensive_test_suite()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
