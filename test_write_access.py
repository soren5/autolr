#!/usr/bin/env python3
"""
Write access test script for cache directory
Tests different file operations that your workload might perform

Usage: python test_write_access.py <directory_path>
"""

import os
import sys
import tempfile
import argparse
from pathlib import Path

def test_write_access(test_dir):
    """Test various write operations in the given directory"""
    
    print(f"\n🔍 Testing write access in: {test_dir}")
    print("=" * 60)
    
    # Get current process info
    print(f"Process UID: {os.getuid()}, GID: {os.getgid()}")
    print(f"Supplementary groups: {os.getgroups()}")
    
    # Convert to Path object for easier handling
    test_path = Path(test_dir)
    
    # Test 1: Directory existence and permissions
    print(f"\n📁 Test 1: Directory check")
    if test_path.exists():
        print(f"✓ Directory exists")
        stat = test_path.stat()
        print(f"  Permissions: {oct(stat.st_mode)[-3:]}")
        print(f"  Owner UID: {stat.st_uid}, GID: {stat.st_gid}")
    else:
        print(f"✗ Directory does not exist")
        return False
    
    # Test 2: Create a simple file
    print(f"\n📝 Test 2: Create file")
    test_file = test_path / "test_write.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("Testing write access\n")
        print(f"✓ Successfully created: {test_file}")
    except PermissionError as e:
        print(f"✗ Permission denied: {e}")
        return False
    except Exception as e:
        print(f"✗ Other error: {e}")
        return False
    
    # Test 3: Read back the file
    print(f"\n📖 Test 3: Read file")
    try:
        with open(test_file, 'r') as f:
            content = f.read().strip()
        print(f"✓ Successfully read: '{content}'")
    except PermissionError as e:
        print(f"✗ Permission denied reading: {e}")
        return False
    
    # Test 4: Append to file
    print(f"\n✏️ Test 4: Append to file")
    try:
        with open(test_file, 'a') as f:
            f.write("Appending more data\n")
        print(f"✓ Successfully appended")
    except PermissionError as e:
        print(f"✗ Permission denied appending: {e}")
        return False
    
    # Test 5: Create subdirectory
    print(f"\n📂 Test 5: Create subdirectory")
    subdir = test_path / "subdir_test"
    try:
        subdir.mkdir(exist_ok=True)
        print(f"✓ Successfully created: {subdir}")
    except PermissionError as e:
        print(f"✗ Permission denied creating directory: {e}")
        return False
    
    # Test 6: Create file in subdirectory
    print(f"\n📄 Test 6: Create file in subdirectory")
    subdir_file = subdir / "nested_file.txt"
    try:
        with open(subdir_file, 'w') as f:
            f.write("Nested file content\n")
        print(f"✓ Successfully created nested file")
    except PermissionError as e:
        print(f"✗ Permission denied creating nested file: {e}")
        return False
    
    # Test 7: Check umask effect
    print(f"\n🔧 Test 7: Check file permissions")
    try:
        stat = subdir_file.stat()
        print(f"  Nested file permissions: {oct(stat.st_mode)[-3:]}")
        print(f"  Owner: {stat.st_uid}:{stat.st_gid}")
    except Exception as e:
        print(f"✗ Could not stat file: {e}")
    
    # Test 8: Use tempfile module
    print(f"\n🔄 Test 8: Temporary file creation")
    try:
        with tempfile.NamedTemporaryFile(dir=test_path, suffix='.tmp', delete=False) as tf:
            tf.write(b"Temporary data\n")
            temp_name = tf.name
        print(f"✓ Successfully created temp file: {temp_name}")
        os.unlink(temp_name)  # Clean up
    except PermissionError as e:
        print(f"✗ Permission denied for temp file: {e}")
        return False
    
    # Test 9: Clean up test files
    print(f"\n🧹 Test 9: Clean up")
    try:
        # Remove test files and directories
        if subdir_file.exists():
            subdir_file.unlink()
        if subdir.exists():
            subdir.rmdir()
        if test_file.exists():
            test_file.unlink()
        print(f"✓ Successfully cleaned up test files")
    except PermissionError as e:
        print(f"✗ Permission denied during cleanup: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! Directory is writable.")
    return True

def main():
    # Get the first argument passed on to the script and use it as the directory to test

    args = sys.argv[1:] if len(sys.argv) > 1 else None
    
    # Determine which directory to test
    if args:
        test_dir = args[0]
        print(f"📌 Testing specified directory: {test_dir}")
    else:
        # Fall back to environment variable or default
        test_dir = os.environ.get('TRANSFORMERS_CACHE', '/home/autolr/models')
        print(f"📌 No directory specified, using: {test_dir}")
        print(f"   (from TRANSFORMERS_CACHE env var or default)")
    
    success = test_write_access(test_dir)
    
    # Optional: Also test NFS directory if we want comparison
    print("\n" + "=" * 60)
    print("🤔 Do you want to test another directory? (y/n)")
    # This part is interactive, so maybe skip in automated environments
    # Instead, we'll just note that you can run again with different path
    
    print("\n💡 Tip: You can test any directory by providing it as an argument:")
    print("   python test_write_access.py /path/to/test")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()