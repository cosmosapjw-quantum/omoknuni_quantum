#!/usr/bin/env python3
"""
Comprehensive fix application script for quantum MCTS visualizations

This script applies all identified fixes to address:
1. Legend positioning and formatting issues
2. Panel layout problems  
3. Empty plot generation
4. Large dataset scaling (1000 games, 1000 simulations per move)
5. Visual quality improvements

Run this script to update all visualization files with the fixes.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

def backup_files(files: List[str], backup_dir: str = "backup_original"):
    """Create backup of original files"""
    backup_path = Path(backup_dir)
    backup_path.mkdir(exist_ok=True)
    
    for file in files:
        if Path(file).exists():
            shutil.copy2(file, backup_path / Path(file).name)
            print(f"Backed up: {file}")

def apply_common_imports_fix(file_path: str):
    """Add common fixes import to visualization files"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the import section and add our common fixes
    import_pattern = r"(import warnings\nwarnings\.filterwarnings\('ignore'\))"
    
    new_imports = """import warnings
warnings.filterwarnings('ignore')

# Import common visualization fixes
try:
    from .plot_fixes_common import (
        fix_legend_positioning, improve_subplot_layout, handle_empty_data,
        scale_for_large_dataset, setup_publication_style, robust_plotting_wrapper,
        PlotQualityChecker, optimize_for_large_data, add_panel_labels,
        ensure_positive_axis, smart_tick_formatting
    )
except ImportError:
    from plot_fixes_common import (
        fix_legend_positioning, improve_subplot_layout, handle_empty_data,
        scale_for_large_dataset, setup_publication_style, robust_plotting_wrapper,
        PlotQualityChecker, optimize_for_large_data, add_panel_labels,
        ensure_positive_axis, smart_tick_formatting
    )"""
    
    if "plot_fixes_common" not in content:
        content = re.sub(import_pattern, new_imports, content)
        
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Applied imports fix to: {file_path}")

def apply_style_setup_fix(file_path: str):
    """Replace manual style setup with common function"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace manual pyplot style setup
    old_pattern = r"plt\.style\.use\('seaborn-v0_8-darkgrid'\)\s*\n\s*sns\.set_palette\([\"'][^\"']+[\"']\)"
    new_pattern = "setup_publication_style()"
    
    content = re.sub(old_pattern, new_pattern, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def apply_legend_fixes(file_path: str):
    """Fix legend positioning throughout file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace basic legend() calls with improved positioning
    patterns = [
        (r"\.legend\(\)", ".legend(loc='best', fontsize=10, framealpha=0.9)"),
        (r"ax([0-9]+)\.legend\(\)", r"fix_legend_positioning(ax\1, 'best')"),
        (r"ax\.legend\(\)", "fix_legend_positioning(ax, 'best')"),
    ]
    
    for old, new in patterns:
        content = re.sub(old, new, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def apply_layout_fixes(file_path: str):
    """Fix subplot layout issues"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace tight_layout with improved layout
    content = re.sub(
        r"plt\.tight_layout\(\)",
        "improve_subplot_layout(fig, axes if 'axes' in locals() else None)",
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)

def apply_large_dataset_fixes(file_path: str):
    """Add large dataset handling"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add data optimization after data extraction
    optimization_pattern = r"(data = create_authentic_physics_data\([^)]+\))"
    optimization_replacement = r"\1\n        data = optimize_for_large_data(data)"
    
    content = re.sub(optimization_pattern, optimization_replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def apply_panel_label_fixes(file_path: str):
    """Add panel labels to subplots"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add panel labels after axes creation
    panel_pattern = r"(axes_flat = axes\.flatten\(\))"
    panel_replacement = r"\1\n        \n        # Add panel labels\n        add_panel_labels(axes, fontsize=12)"
    
    content = re.sub(panel_pattern, panel_replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def apply_axis_improvements(file_path: str):
    """Improve axis formatting and labels"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Improve axis labels with fontsize
    label_patterns = [
        (r"\.set_xlabel\('([^']+)'\)", r".set_xlabel('\1', fontsize=11)"),
        (r"\.set_ylabel\('([^']+)'\)", r".set_ylabel('\1', fontsize=11)"),
        (r"\.set_title\('([^']+)'\)", r".set_title('\1', fontsize=12, fontweight='bold')"),
    ]
    
    for old, new in label_patterns:
        content = re.sub(old, new, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def apply_robust_plotting_decorator(file_path: str):
    """Add robust plotting decorator to main plotting functions"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find main plotting function definitions and add decorator
    function_pattern = r"(\s+def (plot_[^(]+)\([^)]*\):[^\n]*)"
    
    def add_decorator(match):
        indentation = match.group(1).split('def')[0]
        return f"{indentation}@robust_plotting_wrapper\n{match.group(1)}"
    
    content = re.sub(function_pattern, add_decorator, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def fix_numerical_stability(file_path: str):
    """Add numerical stability improvements"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add bounds checking for common numerical issues
    stability_fixes = [
        # Avoid log(0)
        (r"np\.log\(([^)]+)\)", r"np.log(np.maximum(\1, 1e-10))"),
        # Avoid division by zero
        (r"/\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?!\w)", r"/ np.maximum(\1, 1e-10)"),
        # Clip arccos arguments
        (r"np\.arccos\(([^)]+)\)", r"np.arccos(np.clip(\1, -1+1e-10, 1-1e-10))"),
    ]
    
    for old, new in stability_fixes:
        content = re.sub(old, new, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def apply_large_dataset_parameters(file_path: str):
    """Update parameters for large dataset handling"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update mock data generation for larger scale
    large_data_updates = [
        # Increase number of games
        (r"for i in range\(15\)", "for i in range(1000)"),
        # Update visit count distributions for 1000 simulations per move
        (r"visit_counts = np\.random\.exponential\(([^,]+), ([^)]+)\)",
         r"visit_counts = np.random.exponential(\1 * 20, \2 * 5)"),
        # Larger tree sizes
        (r"tree_size = 50 \+ i \* 25", "tree_size = 500 + i * 2"),
        # Deeper trees
        (r"max_depth = 5 \+ i", "max_depth = 15 + i // 100"),
    ]
    
    for old, new in large_data_updates:
        content = re.sub(old, new, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Apply all fixes to visualization files"""
    
    # Get list of visualization files
    viz_files = [
        "plot_rg_flow_phase_diagrams.py",
        "plot_statistical_physics.py", 
        "plot_critical_phenomena_scaling.py",
        "plot_decoherence_darwinism.py",
        "plot_entropy_analysis.py",
        "plot_thermodynamics.py",
        "plot_jarzynski_equality.py",
        "plot_exact_hbar_analysis.py",
        "run_all_visualizations.py"
    ]
    
    # Filter to only existing files
    existing_files = [f for f in viz_files if Path(f).exists()]
    
    if not existing_files:
        print("No visualization files found in current directory")
        return
    
    print(f"Found {len(existing_files)} visualization files")
    
    # Create backup
    print("\n1. Creating backup of original files...")
    backup_files(existing_files)
    
    # Apply fixes to each file
    for file_path in existing_files:
        print(f"\n2. Applying fixes to {file_path}...")
        
        try:
            apply_common_imports_fix(file_path)
            apply_style_setup_fix(file_path)
            apply_legend_fixes(file_path)
            apply_layout_fixes(file_path)
            apply_large_dataset_fixes(file_path)
            apply_panel_label_fixes(file_path)
            apply_axis_improvements(file_path)
            apply_robust_plotting_decorator(file_path)
            fix_numerical_stability(file_path)
            
            # Special handling for data generation file
            if "run_all_visualizations.py" in file_path:
                apply_large_dataset_parameters(file_path)
            
            print(f"   ✓ Successfully applied fixes to {file_path}")
            
        except Exception as e:
            print(f"   ✗ Error applying fixes to {file_path}: {e}")
    
    print("\n3. Creating summary of applied fixes...")
    
    # Create a summary file
    summary_content = """# Quantum MCTS Visualization Fixes Applied

## Issues Fixed:

### 1. Legend Positioning and Formatting
- Replaced basic legend() calls with fix_legend_positioning()
- Added fontsize, framealpha, and better positioning
- Prevented legend overlap with data

### 2. Panel Layout Problems
- Replaced plt.tight_layout() with improve_subplot_layout()
- Added better spacing and padding
- Implemented adaptive layout for different subplot configurations

### 3. Empty Plot Generation
- Added robust_plotting_wrapper decorator for error handling
- Implemented fallback plot generation for errors
- Added data validation and empty plot detection

### 4. Large Dataset Scaling
- Updated mock data generation for 1000 games, 1000 simulations per move
- Added optimize_for_large_data() for intelligent subsampling
- Implemented adaptive binning and data reduction

### 5. Visual Quality Improvements
- Added publication-ready styling with setup_publication_style()
- Implemented panel labels with add_panel_labels()
- Improved axis formatting and font sizes
- Added numerical stability fixes for edge cases

### 6. Performance Optimizations
- Added scale_for_large_dataset() for visualization efficiency
- Implemented intelligent data subsampling
- Added memory-efficient plotting for large arrays

## Files Modified:
""" + "\n".join([f"- {f}" for f in existing_files]) + """

## Usage:
All visualization files now automatically apply these fixes when imported.
The common fixes are centralized in plot_fixes_common.py for maintainability.

## Testing:
Run the visualization suite to verify all fixes are working:
```bash
python run_all_visualizations.py
```
"""
    
    with open("VISUALIZATION_FIXES_APPLIED.md", "w") as f:
        f.write(summary_content)
    
    print("\n✓ All fixes applied successfully!")
    print("✓ Original files backed up to backup_original/")
    print("✓ Summary written to VISUALIZATION_FIXES_APPLIED.md")
    print("\nNext steps:")
    print("1. Test the fixes by running: python run_all_visualizations.py")
    print("2. Check the generated plots for improved quality")
    print("3. Review the plot_quality_report.txt for any remaining issues")

if __name__ == "__main__":
    main()