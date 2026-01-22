"""
Test runner script for XENO Bot
Run this script to execute all tests with coverage reporting
"""
import sys
import subprocess
import argparse


def run_tests(test_path=None, with_coverage=True, verbose=False):
    """Run tests with optional coverage"""
    print("=" * 70)
    print("Running XENO Bot Unit Tests")
    print("=" * 70)
    
    # Build command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add coverage if requested
    if with_coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html",
        ])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add test path
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")
    
    # Show the command being run
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run tests
    result = subprocess.run(cmd)
    
    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("✓ All tests passed!")
        if with_coverage:
            print("Coverage report generated in htmlcov/index.html")
    else:
        print("✗ Some tests failed!")
    print("=" * 70)
    
    return result.returncode


def main():
    """Parse command line arguments and run tests"""
    parser = argparse.ArgumentParser(description="Run XENO Bot tests")
    parser.add_argument(
        "path", 
        nargs="?", 
        default=None,
        help="Path to specific test file or directory (default: all tests)"
    )
    parser.add_argument(
        "--no-coverage", 
        action="store_true",
        help="Run tests without coverage analysis"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tests without running them"
    )
    
    args = parser.parse_args()
    
    # List tests if requested
    if args.list:
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "tests/"]
        subprocess.run(cmd)
        return 0
    
    # Run tests
    return run_tests(
        test_path=args.path,
        with_coverage=not args.no_coverage,
        verbose=args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())