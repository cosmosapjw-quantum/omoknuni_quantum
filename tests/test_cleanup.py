"""Test cleanup operations"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCleanup(unittest.TestCase):
    """Test cleanup operations"""
    
    def setUp(self):
        """Create temporary test directory"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_remove_pycache(self):
        """Test removal of __pycache__ directories"""
        # Create test __pycache__ directory
        pycache_dir = os.path.join(self.test_dir, "__pycache__")
        os.makedirs(pycache_dir)
        
        # Create test .pyc file
        pyc_file = os.path.join(pycache_dir, "test.pyc")
        with open(pyc_file, "w") as f:
            f.write("test")
        
        # Verify it exists
        self.assertTrue(os.path.exists(pycache_dir))
        self.assertTrue(os.path.exists(pyc_file))
        
        # Remove __pycache__
        from cleanup_codebase import remove_pycache
        remove_pycache(self.test_dir)
        
        # Verify it's gone
        self.assertFalse(os.path.exists(pycache_dir))
        
    def test_remove_build_artifacts(self):
        """Test removal of build artifacts"""
        # Create test .so file
        so_file = os.path.join(self.test_dir, "test.so")
        with open(so_file, "w") as f:
            f.write("test")
            
        # Create test .egg-info directory
        egg_dir = os.path.join(self.test_dir, "test.egg-info")
        os.makedirs(egg_dir)
        
        self.assertTrue(os.path.exists(so_file))
        self.assertTrue(os.path.exists(egg_dir))
        
        # Remove build artifacts
        from cleanup_codebase import remove_build_artifacts
        remove_build_artifacts(self.test_dir)
        
        # Verify they're gone
        self.assertFalse(os.path.exists(so_file))
        self.assertFalse(os.path.exists(egg_dir))
        
    def test_organize_imports(self):
        """Test import organization"""
        # Create test Python file with messy imports
        test_file = os.path.join(self.test_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("""import sys
import os
from typing import List
import numpy as np
from dataclasses import dataclass
import torch

def test():
    pass
""")
        
        # Organize imports
        from cleanup_codebase import organize_imports
        organize_imports(test_file)
        
        # Read organized file
        with open(test_file, "r") as f:
            content = f.read()
        
        # Check that standard library imports come first
        lines = content.strip().split("\n")
        self.assertTrue(lines[0].startswith("import os"))
        self.assertTrue(lines[1].startswith("import sys"))
        
    def test_remove_commented_code(self):
        """Test removal of commented code"""
        test_file = os.path.join(self.test_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("""def test():
    # print("debug")
    x = 1
    # old_code = 2
    return x
    
# def old_function():
#     pass
""")
        
        from cleanup_codebase import remove_commented_code
        remove_commented_code(test_file)
        
        with open(test_file, "r") as f:
            content = f.read()
        
        # Check that code comments are removed but function remains
        self.assertNotIn('print("debug")', content)
        self.assertNotIn('old_code', content)
        self.assertNotIn('old_function', content)
        self.assertIn('def test():', content)
        self.assertIn('return x', content)


if __name__ == "__main__":
    unittest.main()