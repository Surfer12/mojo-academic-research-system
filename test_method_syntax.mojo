# Test file to check method parameter syntax in Mojo

from collections import Dict, List

@value
struct TestStruct:
    var name: String
    var count: Int

    fn test_method1(self) -> String:
        """Test method with no mutation."""
        return self.name

    fn test_method2(self inout) -> None:
        """Test method with inout self."""
        self.count += 1

    fn test_method3(self inout, value: Int) -> None:
        """Test method with inout self and parameter."""
        self.count = value

    fn test_method4(self inout, value: Int, name: String) -> None:
        """Test method with inout self and multiple parameters."""
        self.count = value
        self.name = name

struct ComplexStruct:
    var data: Dict[String, String]
    var items: List[String]

    fn add_item(self inout, item: String) -> None:
        """Test method on struct with collections."""
        self.items.append(item)

    fn set_data(self inout, key: String, value: String) -> None:
        """Test method with multiple parameters."""
        self.data[key] = value

fn main():
    """Test the syntax."""
    print("Testing method syntax...")

    # Test @value struct
    var test = TestStruct(name="test", count=0)
    print("Name: " + test.test_method1())

    test.test_method2()
    print("Count after increment: " + String(test.count))

    test.test_method3(5)
    print("Count after set: " + String(test.count))

    test.test_method4(10, "new_name")
    print("Name after update: " + test.name)
    print("Count after update: " + String(test.count))

    # Test complex struct
    var complex = ComplexStruct()
    complex.data = Dict[String, String]()
    complex.items = List[String]()

    complex.add_item("item1")
    complex.set_data("key1", "value1")

    print("Complex struct test complete")
