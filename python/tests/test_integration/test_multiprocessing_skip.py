"""
Multiprocessing tests that are skipped by default to prevent hanging

To run these tests explicitly:
    pytest python/tests/test_integration/test_multiprocessing_skip.py -v

These tests are prone to hanging in CI environments due to:
- Process synchronization issues
- GPU resource contention
- Queue deadlocks
- Shared memory issues
"""

import pytest

# Mark entire module to be skipped
pytestmark = pytest.mark.skip(reason="Multiprocessing tests skipped by default to prevent hanging. Run explicitly if needed.")

# Import the actual test module
from test_multiprocessing import *