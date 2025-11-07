"""
Test runner for MHA Toolbox
============================
Run this to execute all unit tests.
"""

from tests.test_core import run_tests

if __name__ == '__main__':
    print("🧪 MHA Toolbox Unit Test Suite")
    print("="*70)
    run_tests(verbosity=2)
