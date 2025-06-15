#!/usr/bin/env python3
"""
Script to automatically apply common fixes to Mojo source files.
This addresses the most common compilation errors in Mojo code.
"""

import re
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

class MojoFixer:
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.fixes_applied = 0

    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")

    def fix_str_to_string(self, content: str) -> str:
        """Replace str() calls with String() calls."""
        pattern = r'\bstr\s*\('
        replacement = 'String('
        new_content = re.sub(pattern, replacement, content)

        if new_content != content:
            self.fixes_applied += 1
            self.log("Fixed str() to String() conversions")

        return new_content

    def add_copyinit_to_struct(self, content: str) -> str:
        """Add __copyinit__ method to structs that don't have one."""
        struct_pattern = r'struct\s+(\w+):\s*\n((?:(?!struct\s+\w+:).*\n)*)'

        def process_struct(match):
            struct_name = match.group(1)
            struct_body = match.group(2)

            # Check if __copyinit__ already exists
            if '__copyinit__' in struct_body:
                return match.group(0)

            # Find all var declarations
            var_pattern = r'^\s*var\s+(\w+):\s*(.+)$'
            vars_found = re.findall(var_pattern, struct_body, re.MULTILINE)

            if not vars_found:
                return match.group(0)

            # Generate __copyinit__ method
            copyinit_lines = [
                f"\n    fn __copyinit__(inout self, existing: Self):"
            ]

            for var_name, var_type in vars_found:
                copyinit_lines.append(f"        self.{var_name} = existing.{var_name}")

            # Find the last method in the struct
            last_fn_match = None
            for fn_match in re.finditer(r'(fn\s+\w+.*?\n(?:\s{4,}.*\n)*)', struct_body):
                last_fn_match = fn_match

            if last_fn_match:
                # Insert after the last function
                insert_pos = last_fn_match.end()
                new_body = (struct_body[:insert_pos] +
                           '\n'.join(copyinit_lines) + '\n' +
                           struct_body[insert_pos:])
            else:
                # No functions found, append at the end
                new_body = struct_body.rstrip() + '\n' + '\n'.join(copyinit_lines) + '\n'

            self.fixes_applied += 1
            self.log(f"Added __copyinit__ to struct {struct_name}")

            return f"struct {struct_name}:\n{new_body}"

        return re.sub(struct_pattern, process_struct, content, flags=re.MULTILINE)

    def fix_dict_iteration(self, content: str) -> str:
        """Fix dictionary iteration patterns."""
        # Pattern for: for key, value in dict.items():
        pattern = r'for\s+(\w+)\s*,\s*(\w+)\s+in\s+(\w+)\.items\(\):'

        def replace_dict_iter(match):
            key_var = match.group(1)
            value_var = match.group(2)
            dict_var = match.group(3)

            self.fixes_applied += 1
            self.log(f"Fixed dict iteration for {dict_var}")

            return f"""for entry in {dict_var}.items():
        var {key_var} = entry.key
        var {value_var} = entry.value"""

        return re.sub(pattern, replace_dict_iter, content)

    def fix_tensor_import(self, content: str) -> str:
        """Fix tensor import issues."""
        if 'from tensor import Tensor' in content:
            # Comment out the problematic import and add a note
            new_content = content.replace(
                'from tensor import Tensor',
                '# TODO: Fix tensor import - check if Tensor is available in current Mojo version\n# from tensor import Tensor'
            )
            if new_content != content:
                self.fixes_applied += 1
                self.log("Commented out problematic tensor import")
            return new_content
        return content

    def fix_function_signatures(self, content: str) -> str:
        """Ensure inout self is properly formatted in function signatures."""
        # Fix patterns like: fn method_name(inout self, ...
        # Make sure there's proper spacing
        pattern = r'fn\s+(\w+)\s*\(\s*inout\s+self\s*,'
        replacement = r'fn \1(inout self,'

        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            self.fixes_applied += 1
            self.log("Fixed function signature formatting")

        return new_content

    def fix_string_concatenation(self, content: str) -> str:
        """Fix string concatenation issues."""
        # Pattern to find print statements with concatenation
        pattern = r'print\s*\(\s*"([^"]+)"\s*\+\s*str\s*\('

        def replace_concat(match):
            prefix = match.group(1)
            return f'print("{prefix}" + String('

        new_content = re.sub(pattern, replace_concat, content)

        if new_content != content:
            self.fixes_applied += 1
            self.log("Fixed string concatenation in print statements")

        return new_content

    def fix_list_initialization(self, content: str) -> str:
        """Fix List initialization patterns."""
        # Pattern: List[Type](...) should be List[Type]() then append
        pattern = r'(\w+)\s*=\s*List\[(\w+)\]\s*\(((?:[^()]+|\([^)]*\))*)\)'

        def check_and_fix_list_init(match):
            var_name = match.group(1)
            list_type = match.group(2)
            init_content = match.group(3).strip()

            # If there's initialization content, it needs to be fixed
            if init_content and init_content != '':
                # Split by commas (simple approach)
                items = [item.strip() for item in init_content.split(',')]

                result = f"{var_name} = List[{list_type}]()"
                for item in items:
                    if item:
                        result += f"\n    {var_name}.append({item})"

                self.fixes_applied += 1
                self.log(f"Fixed List initialization for {var_name}")
                return result

            return match.group(0)

        return re.sub(pattern, check_and_fix_list_init, content)

    def process_file(self, filepath: Path) -> bool:
        """Process a single Mojo file and apply fixes."""
        try:
            with open(filepath, 'r') as f:
                original_content = f.read()

            content = original_content

            # Apply fixes in order
            content = self.fix_str_to_string(content)
            content = self.add_copyinit_to_struct(content)
            content = self.fix_dict_iteration(content)
            content = self.fix_tensor_import(content)
            content = self.fix_function_signatures(content)
            content = self.fix_string_concatenation(content)
            content = self.fix_list_initialization(content)

            if content != original_content:
                if not self.dry_run:
                    # Create backup
                    backup_path = filepath.with_suffix('.mojo.bak')
                    with open(backup_path, 'w') as f:
                        f.write(original_content)

                    # Write fixed content
                    with open(filepath, 'w') as f:
                        f.write(content)

                    print(f"✓ Fixed {filepath} (backup saved as {backup_path})")
                else:
                    print(f"Would fix {filepath}")

                return True
            else:
                if self.verbose:
                    print(f"No fixes needed for {filepath}")
                return False

        except Exception as e:
            print(f"✗ Error processing {filepath}: {e}")
            return False

    def process_directory(self, directory: Path) -> int:
        """Process all Mojo files in a directory."""
        files_fixed = 0
        mojo_files = list(directory.rglob("*.mojo"))

        if not mojo_files:
            print(f"No .mojo files found in {directory}")
            return 0

        print(f"Found {len(mojo_files)} Mojo files to process")

        for filepath in mojo_files:
            if self.process_file(filepath):
                files_fixed += 1

        return files_fixed

def main():
    parser = argparse.ArgumentParser(
        description="Automatically fix common Mojo compilation errors"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a Mojo file or directory containing Mojo files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information about fixes"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files (not recommended)"
    )

    args = parser.parse_args()

    fixer = MojoFixer(dry_run=args.dry_run, verbose=args.verbose)

    if args.path.is_file():
        if args.path.suffix == '.mojo':
            success = fixer.process_file(args.path)
            print(f"\nTotal fixes applied: {fixer.fixes_applied}")
            sys.exit(0 if success else 1)
        else:
            print(f"Error: {args.path} is not a .mojo file")
            sys.exit(1)
    elif args.path.is_dir():
        files_fixed = fixer.process_directory(args.path)
        print(f"\nFiles fixed: {files_fixed}")
        print(f"Total fixes applied: {fixer.fixes_applied}")
        sys.exit(0 if files_fixed > 0 or args.dry_run else 1)
    else:
        print(f"Error: {args.path} not found")
        sys.exit(1)

if __name__ == "__main__":
    main()
