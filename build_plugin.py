#!/usr/bin/env python3
"""
Build script for MatrixToDiscourseBot maubot plugin.
Creates a .mbp file without requiring mbc to be installed.
"""

import os
import sys
import zipfile
import yaml
import subprocess
from pathlib import Path
import argparse


def run_tests():
    """Run plugin validation tests before building."""
    print("🧪 Running plugin validation tests...")
    try:
        result = subprocess.run([sys.executable, "test_plugin.py"], 
                              capture_output=True, text=True)
        
        # Print the test output
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
        if result.returncode != 0:
            print("❌ Tests failed! Fix the issues before building.")
            return False
        else:
            print("✅ All tests passed!")
            return True
    except FileNotFoundError:
        print("⚠️  test_plugin.py not found, skipping validation tests")
        return True
    except Exception as e:
        print(f"⚠️  Error running tests: {e}")
        return True  # Don't fail build if tests can't run


def load_maubot_yaml():
    """Load and parse maubot.yaml file."""
    plugin_dir = Path("plugin")
    maubot_yaml_path = plugin_dir / "maubot.yaml"
    
    try:
        with open(maubot_yaml_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: plugin/maubot.yaml not found.")
        print("Please run this script from the repository root directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing maubot.yaml: {e}")
        sys.exit(1)


def get_files_to_include(meta):
    """Get list of files to include in the .mbp archive."""
    plugin_dir = Path("plugin")
    files = []
    
    # Always include maubot.yaml
    files.append("maubot.yaml")
    
    # Add all module files
    modules = meta.get("modules", [])
    for module in modules:
        module_path = plugin_dir / module
        
        # Check if it's a package (directory with __init__.py)
        if module_path.is_dir() and (module_path / "__init__.py").exists():
            # Add all Python files in the package
            for py_file in module_path.rglob("*.py"):
                # Store relative to plugin directory
                files.append(str(py_file.relative_to(plugin_dir)))
        # Check if it's a single Python file
        elif (plugin_dir / f"{module}.py").exists():
            files.append(f"{module}.py")
        else:
            print(f"Warning: Module '{module}' not found, skipping...")
    
    # Add extra files if specified
    extra_files = meta.get("extra_files", [])
    for extra_file in extra_files:
        if (plugin_dir / extra_file).exists():
            files.append(extra_file)
        else:
            print(f"Warning: Extra file '{extra_file}' not found, skipping...")
    
    # Add base-config.yaml if it exists and config is enabled (and not already added)
    if meta.get("config", False) and (plugin_dir / "base-config.yaml").exists():
        if "base-config.yaml" not in files:
            files.append("base-config.yaml")
    
    return files


def create_mbp(meta, output_file=None):
    """Create the .mbp file."""
    plugin_id = meta.get("id", "unknown")
    version = meta.get("version", "0.0.0")
    
    if output_file is None:
        # Generate output filename from plugin ID and version
        output_file = f"{plugin_id}-v{version}.mbp"
    
    files_to_include = get_files_to_include(meta)
    plugin_dir = Path("plugin")
    
    print(f"Creating {output_file}...")
    print(f"Including {len(files_to_include)} files:")
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to plugin directory for correct relative paths
        os.chdir(plugin_dir)
        
        with zipfile.ZipFile(f"../{output_file}", "w", zipfile.ZIP_DEFLATED) as mbp:
            for file_path in files_to_include:
                if Path(file_path).exists():
                    print(f"  - {file_path}")
                    mbp.write(file_path)
                else:
                    print(f"  - {file_path} (NOT FOUND)")
    finally:
        # Change back to original directory
        os.chdir(original_dir)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Build maubot plugin .mbp file")
    parser.add_argument("-o", "--output", help="Output file name (default: auto-generated)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--skip-tests", action="store_true", help="Skip validation tests")
    args = parser.parse_args()
    
    print("Building MatrixToDiscourseBot plugin...")
    
    # Check if we're in the right directory
    if not Path("plugin").exists() or not Path("plugin/maubot.yaml").exists():
        print("Error: plugin/maubot.yaml not found.")
        print("Please run this script from the repository root directory.")
        sys.exit(1)
    
    # Run tests first unless skipped
    if not args.skip_tests:
        if not run_tests():
            sys.exit(1)
    
    # Load metadata
    meta = load_maubot_yaml()
    
    if args.verbose:
        print("\nPlugin metadata:")
        print(f"  ID: {meta.get('id', 'unknown')}")
        print(f"  Version: {meta.get('version', 'unknown')}")
        print(f"  License: {meta.get('license', 'unknown')}")
        print(f"  Main class: {meta.get('main_class', 'unknown')}")
        print(f"  Modules: {', '.join(meta.get('modules', []))}")
    
    # Create the .mbp file
    try:
        output_file = create_mbp(meta, args.output)
        
        # Show file info
        file_size = os.path.getsize(output_file)
        print(f"\n✅ Build successful!")
        print(f"Generated plugin file: {output_file}")
        print(f"File size: {file_size:,} bytes")
        print("\nYou can now upload this file to your maubot instance.")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 