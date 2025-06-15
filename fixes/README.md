# Mojo Academic Research System - Compilation Fixes

This directory contains tools and documentation to fix common compilation errors in the Mojo Academic Research System.

## Quick Start

To automatically apply common fixes to all Mojo files:

```bash
# From the project root directory
cd mojo-academic-research-system

# Apply fixes to all Mojo files (creates backups)
python3 fixes/apply_mojo_fixes.py .

# Or preview what would be fixed without making changes
python3 fixes/apply_mojo_fixes.py . --dry-run

# For verbose output
python3 fixes/apply_mojo_fixes.py . --verbose
```

## Common Compilation Errors and Solutions

### 1. Missing `__copyinit__` Methods

**Error**: `struct 'X' does not implement all requirements for 'Copyable'`

The automated script will add `__copyinit__` methods to structs that need them. These are required for structs that will be stored in collections like `List` or `Dict`.

### 2. String Conversion Issues

**Error**: `use of unknown declaration 'str'`

- The script automatically replaces `str()` with `String()`
- In Mojo, use `String()` instead of Python's `str()`

### 3. Dictionary Iteration

**Error**: `cannot unpack value of type 'DictEntry[String, X]' into 2 values`

The script fixes patterns like:
```mojo
# Before (incorrect)
for key, value in my_dict.items():
    print(key)

# After (correct)
for entry in my_dict.items():
    var key = entry.key
    var value = entry.value
    print(key)
```

### 4. Import Issues

**Error**: `package 'tensor' does not contain 'Tensor'`

The script comments out problematic tensor imports. You may need to:
- Check if Tensor is available in your Mojo version
- Use alternative approaches or Python interop

## Manual Fixes

Some issues require manual intervention:

### Function Signatures

Ensure `inout` parameters are properly formatted:
```mojo
# Correct
fn my_method(inout self, param: String) -> Bool:
    # implementation
```

### List Initialization

The script attempts to fix list initialization, but complex cases may need manual adjustment:
```mojo
# Simple initialization
var my_list = List[String]()
my_list.append("item1")
my_list.append("item2")
```

## Testing Your Fixes

After applying fixes:

```bash
# Test individual modules
mojo run pattern_matcher.mojo
mojo run validation_system.mojo
mojo run research_config.mojo
mojo run academic_research_workflow.mojo

# Test the example
mojo run example_usage.mojo

# Or use the CLI
python3 scripts/cli.py example
```

## Backup Files

The script automatically creates `.mojo.bak` backup files. To restore:
```bash
# Restore a single file
mv file.mojo.bak file.mojo

# Restore all files
for f in *.mojo.bak; do mv "$f" "${f%.bak}"; done
```

## Additional Resources

- `mojo_fixes_guide.md` - Detailed explanation of each fix type
- `apply_mojo_fixes.py` - Source code of the automated fixer

## Troubleshooting

If compilation still fails after applying fixes:

1. Check the specific error message
2. Look for patterns in `mojo_fixes_guide.md`
3. Ensure you're using a compatible Mojo version
4. Some advanced features may need manual refactoring

## Known Limitations

- Complex dictionary unpacking may need manual fixes
- Some type inference issues require explicit type annotations
- Advanced generic types might need adjustment
- Python interop code may need special handling

## Contributing

If you discover new fix patterns, please update:
1. The `apply_mojo_fixes.py` script
2. The `mojo_fixes_guide.md` documentation