# Save as run_cognitive_tests.py in your project root
#!/usr/bin/env python
"""Run cognitive memory tests directly without pytest interference"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the test class directly
from tests.test_cognitive_memory_simple import TestCognitiveMemorySimple

# Run tests manually
print("Running Cognitive Memory Tests...")
print("=" * 60)

# Create an instance of the test class
test_instance = TestCognitiveMemorySimple()

# Run each test method
test_methods = [
    'test_init_creates_driver_with_config',
    'test_close_closes_driver', 
    'test_upsert_user_creates_new_user',
    'test_create_memory_creates_node_and_links_to_user',
    'test_write_memory_calls_all_methods'
]

for method_name in test_methods:
    print(f"\nRunning: {method_name}")
    print("-" * 40)
    try:
        test_method = getattr(test_instance, method_name)
        test_method()
        print(f"✓ PASSED: {method_name}")
    except Exception as e:
        print(f"✗ FAILED: {method_name}")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Test run completed!")