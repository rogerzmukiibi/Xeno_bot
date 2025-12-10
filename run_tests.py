"""
Test runner script for XENO Bot
Run this script to execute all tests with coverage reporting
"""
import sys
import subprocess


def run_tests():
    """Run all tests with coverage"""
    print("=" * 70)
    print("Running XENO Bot Unit Tests")
    print("=" * 70)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v"
    ]
    
    result = subprocess.run(cmd)
    
    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("✓ All tests passed!")
        print("Coverage report generated in htmlcov/index.html")
    else:
        print("✗ Some tests failed!")
    print("=" * 70)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())
