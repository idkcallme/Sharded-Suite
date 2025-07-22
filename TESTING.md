# GGUF Shard Suite - Enhanced Testing Infrastructure

This document describes the comprehensive testing infrastructure for the GGUF Shard Suite, including performance monitoring, regression detection, and multiple testing modes.

## Testing Files Overview

### Core Test Files

- **`test_real_model.py`** - Original basic real model workflow test
- **`test_real_model_enhanced.py`** - Enhanced test suite with performance monitoring and regression detection
- **`test_runner.py`** - Interactive test runner with multiple test suites
- **`test_config.json`** - Configuration file for test settings and thresholds

### Test Execution Logs

- **`test_execution.log`** - Detailed logging of test execution
- **`test_baselines.json`** - Performance baselines for regression detection
- **`benchmark_results.json`** - Benchmark performance results

## Available Test Suites

### 1. Basic Test Suite
```bash
python test_runner.py --suite basic
```
Runs the original workflow test with basic validation.

### 2. Enhanced Test Suite
```bash
python test_runner.py --suite enhanced
```
Features:
- Real-time CPU and memory monitoring
- Performance regression detection
- Comprehensive error handling and logging
- File integrity verification
- Memory stress testing for large models

### 3. Stress Test Suite
```bash
python test_runner.py --suite stress
```
- Runs multiple iterations with the largest available model
- Tests system stability under repeated load
- Requires 80% success rate to pass

### 4. Regression Test Suite
```bash
python test_runner.py --suite regression
```
- Tracks performance baselines over time
- Detects performance degradations (default: 15% threshold)
- Maintains historical performance data

### 5. Matrix Test Suite
```bash
python test_runner.py --suite matrix
```
- Tests all available GGUF models in the directory
- Provides comprehensive compatibility testing
- Generates per-model pass/fail matrix

### 6. Benchmark Suite
```bash
python test_runner.py --suite benchmark
```
- Measures throughput across all models
- Runs multiple iterations for consistent results
- Saves results to `benchmark_results.json`
- Ranks models by processing speed

### 7. Complete Test Suite
```bash
python test_runner.py --suite all
```
Runs all test suites in sequence with comprehensive reporting.

## Interactive Mode

Launch the interactive menu:
```bash
python test_runner.py --interactive
```

This provides a menu-driven interface to select and run specific test suites.

## Enhanced Features

### Performance Monitoring
- Real-time CPU usage tracking
- Memory consumption monitoring  
- Peak memory detection
- Throughput calculation (MB/s)

### Regression Detection
- Automatic baseline establishment
- Configurable regression thresholds
- Historical performance tracking
- Warning alerts for performance degradation

### Comprehensive Logging
- Detailed execution logs
- Error tracking with stack traces
- Performance metrics logging
- Test result archival

### Error Handling
- Context managers for test steps
- Graceful failure handling
- Automatic cleanup of test files
- Resource leak prevention

### File Integrity Checking
- GGUF header validation
- Shard map structure verification
- Required field validation
- Corrupted file detection

## Configuration

Edit `test_config.json` to customize:

```json
{
  "test_config": {
    "performance": {
      "regression_threshold": 0.15,    // 15% performance degradation threshold
      "benchmark_iterations": 3,        // Benchmark repetitions
      "stress_test_iterations": 5,      // Stress test repetitions  
      "memory_monitoring_interval": 0.1 // Monitor every 100ms
    },
    "validation": {
      "min_throughput_mb_s": 10.0,     // Minimum acceptable throughput
      "max_memory_usage_mb": 8192       // Maximum memory usage limit
    }
  }
}
```

## Command Line Options

### Enhanced Test Suite Options
```bash
python test_real_model_enhanced.py --model <specific_model.gguf>  # Test specific model
python test_real_model_enhanced.py --no-regression               # Skip regression testing
python test_real_model_enhanced.py --log-level DEBUG             # Set logging level
```

### Test Runner Options
```bash
python test_runner.py --list                    # List available test suites
python test_runner.py --interactive            # Interactive mode
python test_runner.py --suite <suite_name>     # Run specific suite
```

## Performance Metrics

### Tracked Metrics
- **Sharding Throughput**: MB/s processing speed
- **Memory Efficiency**: Peak memory usage during operations  
- **CPU Utilization**: Average CPU usage during processing
- **Delta Compression**: Compression ratio for incremental updates
- **Storage Efficiency**: Space saved through sharding

### Baseline Management
- Baselines are automatically established on first run
- Running averages maintained over multiple test runs
- Regression alerts when performance degrades beyond threshold
- Historical data preserved in `test_baselines.json`

## CI/CD Integration

The test suites are designed for integration with continuous integration:

```yaml
# Example GitHub Actions integration
- name: Run GGUF Shard Suite Tests
  run: |
    python test_runner.py --suite enhanced
    python test_runner.py --suite regression
```

## Troubleshooting

### Common Issues

1. **Missing psutil**: Install with `pip install psutil`
2. **No GGUF files found**: Ensure test model files are in the current directory
3. **Permission errors**: Check file write permissions for log files
4. **Memory errors**: Reduce concurrent operations or test with smaller models

### Debug Mode
```bash
python test_real_model_enhanced.py --log-level DEBUG
```

### Log Files
- Check `test_execution.log` for detailed execution information
- Performance metrics are logged with timestamps
- Error stack traces included for debugging

## Best Practices

1. **Regular Baseline Updates**: Run tests regularly to maintain accurate baselines
2. **Resource Monitoring**: Monitor system resources during large model testing
3. **Test Data Management**: Keep test models in a dedicated directory
4. **Performance Tracking**: Review benchmark results for performance trends
5. **Error Analysis**: Check logs after failures for root cause analysis

## Performance Expectations

### Typical Results
- **Small models (< 100MB)**: 50-200 MB/s throughput
- **Medium models (100MB-1GB)**: 100-150 MB/s throughput  
- **Large models (> 1GB)**: 80-120 MB/s throughput
- **Memory overhead**: ~10-20% of model size
- **Delta compression**: 99.9%+ efficiency for small changes

### Regression Thresholds
- **Throughput degradation**: > 15% slower than baseline
- **Memory usage increase**: > 20% more than baseline
- **Error rate increase**: > 5% more failures than baseline
