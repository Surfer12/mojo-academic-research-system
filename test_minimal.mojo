# Minimal test file to isolate and fix compilation issues
from collections import Dict, List

struct SimplePattern:
    var pattern_id: String
    var weight: Float32

    fn __init__(inout self, pattern_id: String, weight: Float32 = 1.0):
        self.pattern_id = pattern_id
        self.weight = weight

struct SimpleConfig:
    var name: String
    var enabled: Bool

    fn __init__(inout self, name: String):
        self.name = name
        self.enabled = True

fn test_basic_functionality():
    """Test basic struct creation and usage"""
    print("=== Testing Basic Functionality ===")

    # Test simple pattern creation
    var pattern = SimplePattern("test_pattern", 0.8)
    print("Created pattern: " + pattern.pattern_id)
    print("Pattern weight: " + String(pattern.weight))

    # Test simple config creation
    var config = SimpleConfig("test_config")
    print("Created config: " + config.name)
    print("Config enabled: " + String(config.enabled))

    # Test collections
    var patterns = List[SimplePattern]()
    patterns.append(pattern)
    print("Added pattern to list, size: " + String(len(patterns)))

    var configs = Dict[String, SimpleConfig]()
    configs["test"] = config
    print("Added config to dict")

    print("Basic functionality test completed successfully!")

fn main():
    """Main test function"""
    print("Starting minimal compilation test...")
    test_basic_functionality()
    print("All tests passed!")
