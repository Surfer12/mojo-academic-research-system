# Mojo Compilation Fixes Guide

This guide addresses the common compilation errors in the Mojo Academic Research System.

## 1. Missing `__copyinit__` for Copyable Structs

**Error**: `struct 'X' does not implement all requirements for 'Copyable'`

**Fix**: Add a `__copyinit__` method to each struct that needs to be stored in collections:

```mojo
struct EthicalGuidelinesConfig:
    var guideline_name: String
    var requirements: List[String]
    var mandatory: Bool
    var review_process: String
    var documentation_needed: List[String]
    
    fn __init__(inout self, name: String, mandatory: Bool = True):
        self.guideline_name = name
        self.requirements = List[String]()
        self.mandatory = mandatory
        self.review_process = "standard"
        self.documentation_needed = List[String]()
    
    # Add this method:
    fn __copyinit__(inout self, existing: Self):
        self.guideline_name = existing.guideline_name
        self.requirements = existing.requirements
        self.mandatory = existing.mandatory
        self.review_process = existing.review_process
        self.documentation_needed = existing.documentation_needed
```

## 2. Function Parameter Issues

**Error**: `expected ')' in argument list`

**Fix**: Ensure `inout` is properly placed:

```mojo
# Wrong:
fn __init__(inout self, name: String):

# Correct:
fn __init__(inout self, name: String):
    # implementation

# For methods that modify self:
fn validate_statistics(inout self, paper_id: String, paper_content: String) -> StatisticalValidation:
    # implementation
```

## 3. Import Issues

**Error**: `package 'tensor' does not contain 'Tensor'`

**Fix**: Update imports to use the correct module names:

```mojo
# Old (incorrect):
from tensor import Tensor

# New (correct):
from tensor import Tensor
# OR if Tensor is in a different module:
from python import Python
# Then use Python interop for tensor operations
```

## 4. String Conversion Issues

**Error**: `use of unknown declaration 'str'`

**Fix**: In Mojo, use `String()` constructor or string formatting:

```mojo
# Wrong:
print("Found " + str(len(matches)) + " pattern matches")

# Correct:
print("Found " + String(len(matches)) + " pattern matches")

# Or use formatting:
print("Found {} pattern matches".format(len(matches)))
```

## 5. Dict Iteration Issues

**Error**: `cannot unpack value of type 'DictEntry[String, X]' into 2 values`

**Fix**: Use proper Dict iteration methods:

```mojo
# Wrong:
for key, value in my_dict.items():
    print(key + ": " + value)

# Correct:
for entry in my_dict.items():
    print(entry.key + ": " + entry.value)

# Or use keys() and access values:
for key in my_dict.keys():
    var value = my_dict[key]
    print(key + ": " + value)
```

## 6. List Initialization

**Fix**: Ensure proper List initialization:

```mojo
# Initialize empty list:
var my_list = List[String]()

# Initialize with values:
var my_list = List[String]()
my_list.append("value1")
my_list.append("value2")
```

## 7. Struct Member Access

**Error**: `'X' value has no attribute 'Y'`

**Fix**: Ensure the struct is properly initialized and the attribute exists:

```mojo
# Make sure the struct is defined with all needed fields:
struct PatternMatcher:
    var signatures: Dict[String, ResearchSignature]
    var pattern_cache: Dict[String, List[PatternMatch]]
    
    fn __init__(inout self):
        self.signatures = Dict[String, ResearchSignature]()
        self.pattern_cache = Dict[String, List[PatternMatch]]()
```

## 8. Complete Example Fix

Here's a complete example of a fixed struct:

```mojo
from collections import Dict, List

struct ResearchPattern:
    var pattern_name: String
    var pattern_regex: String
    var weight: Float64
    var context_required: Bool
    
    fn __init__(inout self, name: String, regex: String, 
                weight: Float64 = 1.0, context_required: Bool = False):
        self.pattern_name = name
        self.pattern_regex = regex
        self.weight = weight
        self.context_required = context_required
    
    fn __copyinit__(inout self, existing: Self):
        self.pattern_name = existing.pattern_name
        self.pattern_regex = existing.pattern_regex
        self.weight = existing.weight
        self.context_required = existing.context_required
```

## 9. Priority Fixes

Fix in this order:
1. Add `__copyinit__` to all structs
2. Fix function signatures (add proper `inout` placement)
3. Fix string conversions (replace `str()` with `String()`)
4. Fix Dict iterations
5. Fix imports

## 10. Testing Your Fixes

After applying fixes, test incrementally:

```bash
# Test individual modules first:
mojo run pattern_matcher.mojo
mojo run validation_system.mojo
mojo run research_config.mojo

# Then test the main modules:
mojo run academic_research_workflow.mojo

# Finally test the example:
mojo run example_usage.mojo
```

## Common Patterns to Remember

1. **Always implement `__copyinit__` for structs used in collections**
2. **Use `String()` instead of `str()` for conversions**
3. **Use `inout self` for methods that modify the struct**
4. **Initialize all collections in `__init__`**
5. **Use proper Dict iteration with `.items()`, `.keys()`, or `.values()`**