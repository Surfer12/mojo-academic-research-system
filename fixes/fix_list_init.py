#!/usr/bin/env python3
"""
Quick fix for List initialization issues where variable prefixes are missing.
This fixes patterns where List.append() calls are missing the variable prefix.
"""

import re
import sys
from pathlib import Path

def fix_list_append_prefix(content: str) -> str:
    """Fix List append calls that are missing their variable prefix."""

    # Pattern to find List initialization followed by append calls
    # This matches: var_name.field = List[Type]()
    # followed by: field.append(...)
    pattern = r'(\w+)\.(\w+)\s*=\s*List\[(\w+)\]\(\)\s*\n(\s*)(\2\.append\(.*?\)(?:\s*\n\s*\2\.append\(.*?\))*)'

    def fix_append_calls(match):
        var_prefix = match.group(1)
        field_name = match.group(2)
        list_type = match.group(3)
        indent = match.group(4)
        append_calls = match.group(5)

        # Replace field.append with var_prefix.field.append
        fixed_appends = re.sub(
            rf'\b{field_name}\.append\(',
            f'{var_prefix}.{field_name}.append(',
            append_calls
        )

        return f'{var_prefix}.{field_name} = List[{list_type}]()\n{indent}{fixed_appends}'

    # Apply the fix
    fixed_content = re.sub(pattern, fix_append_calls, content, flags=re.MULTILINE | re.DOTALL)

    # Also fix standalone cases where just the variable name is used
    # Pattern: var_name = List[Type]()
    # followed by: var_name.append(...) without prefix
    standalone_pattern = r'^(\s*)(\w+)\s*=\s*List\[(\w+)\]\(\)\s*\n(?=\1(?!\w+\.)\w+\.append)'

    lines = fixed_content.split('\n')
    fixed_lines = []
    current_var = None
    current_indent = None

    for i, line in enumerate(lines):
        # Check if this is a List initialization
        init_match = re.match(r'^(\s*)(\w+)\s*=\s*List\[(\w+)\]\(\)\s*$', line)
        if init_match:
            current_indent = init_match.group(1)
            current_var = init_match.group(2)
            fixed_lines.append(line)
            continue

        # Check if this is an append call that needs fixing
        if current_var and line.strip():
            append_match = re.match(rf'^(\s*)(\w+)\.append\((.*)\)$', line)
            if append_match:
                indent = append_match.group(1)
                var_name = append_match.group(2)
                args = append_match.group(3)

                # If the variable name matches a List field name and doesn't have a prefix
                if var_name != current_var and '.' not in var_name:
                    # This might be a field append that needs the object prefix
                    # Check the context to determine if this needs fixing
                    if len(indent) > len(current_indent):
                        # Skip this fix - it's handled by the first pattern
                        fixed_lines.append(line)
                        continue

                # Reset current_var if we've moved to a different context
                if len(indent) <= len(current_indent):
                    current_var = None

            # Reset if we hit a non-append line
            elif not line.strip().startswith('#'):
                current_var = None

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def process_file(filepath: Path) -> bool:
    """Process a single file and fix List initialization issues."""
    try:
        with open(filepath, 'r') as f:
            original_content = f.read()

        fixed_content = fix_list_append_prefix(original_content)

        if fixed_content != original_content:
            # Save the fixed content
            with open(filepath, 'w') as f:
                f.write(fixed_content)

            print(f"✓ Fixed List initialization in {filepath}")
            return True
        else:
            print(f"No List initialization fixes needed in {filepath}")
            return False

    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_list_init.py <file_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        if path.suffix == '.mojo':
            success = process_file(path)
            sys.exit(0 if success else 1)
        else:
            print(f"Error: {path} is not a .mojo file")
            sys.exit(1)
    elif path.is_dir():
        mojo_files = list(path.rglob("*.mojo"))
        if not mojo_files:
            print(f"No .mojo files found in {path}")
            sys.exit(1)

        fixed_count = 0
        for filepath in mojo_files:
            if process_file(filepath):
                fixed_count += 1

        print(f"\nTotal files fixed: {fixed_count}")
        sys.exit(0 if fixed_count > 0 else 1)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)

if __name__ == "__main__":
    main()
