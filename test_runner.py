#!/usr/bin/env python3
"""
Interactive Test Runner

Provides a CLI interface for running specific test suites
with various testing modes and configurations.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Callable, Dict, Any
import subprocess
import json

class TestRunner:
    """Interactive test runner with menu system"""
    
    def __init__(self):
        self.test_suites = {
            "basic": ("Basic workflow test", self.run_basic),
            "enhanced": ("Enhanced test with performance monitoring", self.run_enhanced),
            "stress": ("Stress testing with large files", self.run_stress),
            "regression": ("Regression detection tests", self.run_regression),
            "matrix": ("Multi-model matrix tests", self.run_matrix),
            "benchmark": ("Performance benchmark suite", self.run_benchmark),
            "all": ("Run all test suites", self.run_all)
        }
        
    def show_menu(self):
        """Display interactive menu"""
        print("\nGGUF Shard Suite Test Runner")
        print("=" * 50)
        
        for key, (description, _) in self.test_suites.items():
            print(f"{key:12} - {description}")
            
        print("\nSelect test suite (or 'quit' to exit):")
        
    def run_interactive(self):
        """Run in interactive mode"""
        while True:
            self.show_menu()
            choice = input("> ").strip().lower()
            
            if choice == 'quit':
                break
            elif choice in self.test_suites:
                _, test_func = self.test_suites[choice]
                print(f"\nRunning {choice} test suite...")
                print("-" * 50)
                start_time = time.time()
                
                try:
                    success = test_func()
                    duration = time.time() - start_time
                    status = "PASSED" if success else "FAILED"
                    print(f"\nTest suite '{choice}' {status} in {duration:.2f}s")
                except KeyboardInterrupt:
                    print(f"\nTest suite '{choice}' interrupted by user")
                except Exception as e:
                    print(f"\nTest suite '{choice}' failed with error: {e}")
            else:
                print(f"Unknown option: {choice}")

    def run_basic(self) -> bool:
        """Run basic real model workflow test"""
        return subprocess.run([
            sys.executable, "test_real_model.py"
        ]).returncode == 0

    def run_enhanced(self) -> bool:
        """Run enhanced test with performance monitoring"""
        return subprocess.run([
            sys.executable, "test_real_model_enhanced.py"
        ]).returncode == 0

    def run_stress(self) -> bool:
        """Run stress tests with various conditions"""
        print("Running stress tests...")
        
        # Find the largest available model
        gguf_files = list(Path('.').glob('*.gguf'))
        gguf_files = [f for f in gguf_files if not f.name.startswith('core')]
        
        if not gguf_files:
            print("No GGUF files found for stress testing")
            return False
        
        largest_file = max(gguf_files, key=lambda f: f.stat().st_size)
        print(f"Using largest model for stress test: {largest_file.name}")
        
        # Run multiple iterations
        success_count = 0
        iterations = 5
        
        for i in range(iterations):
            print(f"\nStress test iteration {i+1}/{iterations}")
            result = subprocess.run([
                sys.executable, "test_real_model_enhanced.py", 
                "--model", str(largest_file)
            ])
            
            if result.returncode == 0:
                success_count += 1
            else:
                print(f"Iteration {i+1} failed")
        
        success_rate = success_count / iterations
        print(f"\nStress test results: {success_count}/{iterations} passed ({success_rate:.1%})")
        
        return success_rate >= 0.8  # 80% success rate required

    def run_regression(self) -> bool:
        """Run regression detection tests"""
        print("Running regression detection...")
        
        # Run enhanced test which includes regression checking
        return subprocess.run([
            sys.executable, "test_real_model_enhanced.py"
        ]).returncode == 0

    def run_matrix(self) -> bool:
        """Run tests across multiple model files"""
        print("Running matrix tests across all available models...")
        
        gguf_files = list(Path('.').glob('*.gguf'))
        gguf_files = [f for f in gguf_files if not f.name.startswith('core')]
        
        if not gguf_files:
            print("No GGUF files found for matrix testing")
            return False
        
        results = {}
        overall_success = True
        
        for model_file in gguf_files:
            print(f"\nTesting model: {model_file.name}")
            result = subprocess.run([
                sys.executable, "test_real_model_enhanced.py", 
                "--model", str(model_file)
            ])
            
            results[model_file.name] = result.returncode == 0
            if result.returncode != 0:
                overall_success = False
        
        # Print matrix results
        print("\nMatrix test results:")
        print("-" * 40)
        for model, success in results.items():
            status = "PASS" if success else "FAIL"
            print(f"{model:30} {status}")
        
        return overall_success

    def run_benchmark(self) -> bool:
        """Run performance benchmark suite"""
        print("Running performance benchmarks...")
        
        gguf_files = list(Path('.').glob('*.gguf'))
        gguf_files = [f for f in gguf_files if not f.name.startswith('core')]
        
        if not gguf_files:
            print("No GGUF files found for benchmarking")
            return False
        
        benchmark_results = {}
        
        for model_file in gguf_files:
            print(f"\nBenchmarking: {model_file.name}")
            
            # Run multiple iterations for consistent results
            throughputs = []
            iterations = 3
            
            for i in range(iterations):
                start_time = time.time()
                result = subprocess.run([
                    sys.executable, "forge/model_sharding_tool.py", 
                    "shard", str(model_file)
                ], capture_output=True)
                
                if result.returncode == 0:
                    duration = time.time() - start_time
                    file_size = model_file.stat().st_size
                    throughput = (file_size / (1024*1024)) / duration
                    throughputs.append(throughput)
                    
                    # Clean up
                    for cleanup_file in ["core.gguf", "core.sgmap"]:
                        cleanup_path = Path(cleanup_file)
                        if cleanup_path.exists():
                            cleanup_path.unlink()
                
            if throughputs:
                avg_throughput = sum(throughputs) / len(throughputs)
                benchmark_results[model_file.name] = avg_throughput
        
        # Display benchmark results
        print("\nBenchmark Results:")
        print("=" * 50)
        print(f"{'Model':<30} {'Throughput (MB/s)':<15}")
        print("-" * 45)
        
        for model, throughput in sorted(benchmark_results.items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"{model:<30} {throughput:>14.2f}")
        
        # Save results to file
        with open("benchmark_results.json", "w") as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "results": benchmark_results
            }, f, indent=2)
        
        print(f"\nBenchmark results saved to benchmark_results.json")
        
        return len(benchmark_results) > 0

    def run_all(self) -> bool:
        """Run all test suites"""
        suite_order = ["basic", "enhanced", "matrix", "benchmark", "stress", "regression"]
        results = {}
        
        print("Running complete test suite...")
        print("=" * 60)
        
        for suite_name in suite_order:
            if suite_name in self.test_suites:
                description, test_func = self.test_suites[suite_name]
                print(f"\nRunning {suite_name}: {description}")
                print("-" * 50)
                
                start_time = time.time()
                try:
                    success = test_func()
                    duration = time.time() - start_time
                    results[suite_name] = {
                        "success": success,
                        "duration": duration
                    }
                    
                    status = "PASSED" if success else "FAILED"
                    print(f"{suite_name} {status} in {duration:.2f}s")
                    
                except Exception as e:
                    results[suite_name] = {
                        "success": False,
                        "duration": time.time() - start_time,
                        "error": str(e)
                    }
                    print(f"{suite_name} FAILED with error: {e}")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("COMPLETE TEST SUITE SUMMARY")
        print("=" * 60)
        
        total_duration = sum(r["duration"] for r in results.values())
        passed = sum(1 for r in results.values() if r["success"])
        total = len(results)
        
        for suite_name, result in results.items():
            status = "PASS" if result["success"] else "FAIL"
            duration = result["duration"]
            print(f"{suite_name:12} {status:4} ({duration:6.2f}s)")
        
        print("-" * 60)
        print(f"Total: {passed}/{total} passed in {total_duration:.2f}s")
        
        return passed == total

def main():
    parser = argparse.ArgumentParser(description="GGUF Shard Suite Test Runner")
    parser.add_argument("--suite", choices=["basic", "enhanced", "stress", "regression", 
                                           "matrix", "benchmark", "all"])
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list", action="store_true", help="List available test suites")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.list:
        print("Available test suites:")
        for key, (description, _) in runner.test_suites.items():
            print(f"  {key:12} - {description}")
        return True
    
    if args.interactive:
        runner.run_interactive()
        return True
    elif args.suite:
        if args.suite in runner.test_suites:
            _, test_func = runner.test_suites[args.suite]
            return test_func()
        else:
            print(f"Unknown test suite: {args.suite}")
            return False
    else:
        parser.print_help()
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
