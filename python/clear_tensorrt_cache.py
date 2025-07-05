#!/usr/bin/env python3
"""Clear TensorRT cache to force recompilation with dynamic batch sizes"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_tensorrt_cache():
    """Clear all TensorRT cached engines"""
    cache_dirs = [
        Path("/tmp/omoknuni_tensorrt_cache"),
        Path.home() / ".cache" / "omoknuni_tensorrt",
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            logger.info(f"Clearing TensorRT cache at: {cache_dir}")
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared {cache_dir}")
        else:
            logger.info(f"Cache directory not found: {cache_dir}")
    
    logger.info("TensorRT cache cleared. Engines will be recompiled on next run.")

if __name__ == "__main__":
    clear_tensorrt_cache()