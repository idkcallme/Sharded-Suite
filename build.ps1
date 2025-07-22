# GGUF Shard Suite Build Script for Windows
param(
    [string]$Target = "build",
    [switch]$EnableCUDA = $false,
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

# Check if we're in the project root
if (-not (Test-Path "CMakeLists.txt")) {
    Write-Error "CMakeLists.txt not found. Please run this script from the project root."
    exit 1
}

# Create build directory
$BuildDir = "build"
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

switch ($Target.ToLower()) {
    "clean" {
        Write-Info "Cleaning build directory..."
        if (Test-Path $BuildDir) {
            Remove-Item -Recurse -Force $BuildDir
            Write-Info "Build directory cleaned."
        }
        else {
            Write-Info "Build directory doesn't exist, nothing to clean."
        }
    }
    
    "build" {
        Write-Info "Building GGUF Shard Suite..."
        
        # Configure CMake
        $CMakeArgs = @(
            "-S", ".",
            "-B", $BuildDir,
            "-DCMAKE_BUILD_TYPE=$BuildType"
        )
        
        if ($EnableCUDA) {
            Write-Info "CUDA support enabled."
            $CMakeArgs += "-DENABLE_CUDA=ON"
        }
        
        Write-Info "Configuring CMake..."
        & cmake @CMakeArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "CMake configuration failed."
            exit 1
        }
        
        # Build
        Write-Info "Building project..."
        & cmake --build $BuildDir --config $BuildType
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Build failed."
            exit 1
        }
        
        Write-Info "Build completed successfully."
    }
    
    "test" {
        Write-Info "Running tests..."
        
        # Check if Python is available
        try {
            $PythonVersion = & python --version 2>&1
            Write-Info "Using Python: $PythonVersion"
        }
        catch {
            Write-Error "Python not found. Please install Python 3.6+ and add it to PATH."
            exit 1
        }
        
        # Install Python dependencies if requirements.txt exists
        if (Test-Path "requirements.txt") {
            Write-Info "Installing Python dependencies..."
            & python -m pip install -r requirements.txt
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Failed to install some Python dependencies, continuing anyway..."
            }
        }
        
        # Run Python test suite
        if (Test-Path "tests\test_suite.py") {
            Write-Info "Running Python test suite..."
            & python tests\test_suite.py
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Python tests failed."
                exit 1
            }
        }
        
        # Run C++ tests if they exist
        $CppTestExecutable = "$BuildDir\$BuildType\gguf_shard_test.exe"
        if (Test-Path $CppTestExecutable) {
            Write-Info "Running C++ tests..."
            & $CppTestExecutable
            if ($LASTEXITCODE -ne 0) {
                Write-Error "C++ tests failed."
                exit 1
            }
        }
        else {
            Write-Info "No C++ test executable found, skipping C++ tests."
        }
        
        Write-Info "All tests passed."
    }
    
    "install" {
        Write-Info "Installing GGUF Shard Suite..."
        
        # Install Python dependencies
        if (Test-Path "requirements.txt") {
            Write-Info "Installing Python dependencies..."
            & python -m pip install -r requirements.txt
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to install Python dependencies."
                exit 1
            }
        }
        
        # Install C++ components
        if (Test-Path $BuildDir) {
            & cmake --install $BuildDir
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to install C++ components."
                exit 1
            }
        }
        
        Write-Info "Installation completed."
    }
    
    default {
        Write-Error "Unknown target: $Target"
        Write-Info "Available targets: clean, build, test, install"
        exit 1
    }
}
