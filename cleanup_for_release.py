#!/usr/bin/env python3
"""
Clean project for public release
Removes personal information and build artifacts
"""

import os
import shutil
from pathlib import Path

def clean_project():
    """Remove personal information from project"""
    
    print("Starting project cleanup for public release...")
    
    # Folders to completely remove
    folders_to_remove = [
        'build',
        '.vscode', 
        '.vs',
        '.github',
        '__pycache__',
        'venv',
        'env',
        '.idea',
        'logs',
        'log'
    ]
    
    # Files to remove
    files_to_remove = [
        'CMakeCache.txt',
        '.DS_Store',
        'Thumbs.db',
        'desktop.ini',
        '.env'
    ]
    
    # Extensions to remove
    extensions_to_remove = [
        '.vcxproj',
        '.vcxproj.filters', 
        '.sln',
        '.user',
        '.suo',
        '.pyc',
        '.pyo',
        '.pyd',
        '.tmp',
        '.temp',
        '.log'
    ]
    
    removed_count = 0
    
    # Remove folders
    for folder in folders_to_remove:
        if os.path.exists(folder):
            print(f"Removing folder: {folder}")
            try:
                shutil.rmtree(folder, ignore_errors=True)
                removed_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {folder}: {e}")
    
    # Remove files
    for file in files_to_remove:
        if os.path.exists(file):
            print(f"Removing file: {file}")
            try:
                os.remove(file)
                removed_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {file}: {e}")
    
    # Remove by extension recursively
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.endswith(ext) for ext in extensions_to_remove):
                file_path = os.path.join(root, file)
                print(f"Removing: {file_path}")
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
    
    print(f"\nCleanup complete! Removed {removed_count} items.")
    print("Project is now ready for public release!")
    print("\nRemaining core files:")
    
    # Show what's left
    core_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if not file.startswith('.') and file != 'cleanup_for_release.py':
                rel_path = os.path.relpath(os.path.join(root, file), '.')
                core_files.append(rel_path)
    
    for file in sorted(core_files):
        print(f"  {file}")

if __name__ == "__main__":
    clean_project()
