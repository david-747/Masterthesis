# Context.py (Modified to use Enums for dimensions)

import itertools
from enum import Enum

class Season(Enum):
    """Enumeration for specific seasons."""
    LOW = "low"
    MID = "mid"
    HIGH = "high"

    def __str__(self):
        return self.value # User-friendly string representation

class CustomerType(Enum):
    """Enumeration for specific customer types."""
    NEW = "new"
    RECURRING = "recurring"

    def __str__(self):
        return self.value

class Commodity(Enum):
    """Enumeration for specific commodity types."""
    ELECTRONICS = "electronics"
    APPAREL = "apparel"
    # Add more as needed

    def __str__(self):
        return self.value

class CommodityValue(Enum):
    """Enumeration for specific commodity value types."""
    HIGH = "high"
    LOW = "low"

    def __str__(self):
        return self.value

class Context:
    """
    Represents a multi-domain context for pricing decisions.
    It's designed to be hashable and comparable and uses Enums for dimensions.
    """
    def __init__(self, season: Season, customer_type: CustomerType, commodity_value: CommodityValue):
        """
        Initializes a context with specific enum members for each dimension.

        Args:
            season (Season): An instance of the Season Enum.
            customer_type (CustomerType): An instance of the CustomerType Enum.
            commodity (Commodity): An instance of the Commodity Enum.
        """
        if not isinstance(season, Season):
            raise TypeError("season must be an instance of Season Enum.")
        if not isinstance(customer_type, CustomerType):
            raise TypeError("customer_type must be an instance of CustomerType Enum.")
        if not isinstance(commodity_value, CommodityValue):
            raise TypeError("commodity must be an instance of CommodityValue Enum.")

        self.season = season
        self.customer_type = customer_type
        self.commodity_value = commodity_value

        # Pre-compute the tuple for hashing and equality. Enum members are hashable.
        self._key = (self.season, self.customer_type, self.commodity_value)

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        if isinstance(other, Context):
            return self._key == other._key
        return False

    def __repr__(self):
        return (f"Context(season={self.season!r}, " # Use !r for Enum's repr
                f"customer_type={self.customer_type!r}, "
                f"commodity={self.commodity_value!r})")

    def __str__(self):
        return (f"Context(Season: {self.season}, " # Relies on Enum's __str__
                f"Customer: {self.customer_type}, "
                f"Commodity: {self.commodity_value})")

    def to_safe_str(self) -> str:
        """Returns a string representation safe for use in filenames or variable names."""
        return f"season_{self.season}_customer_{self.customer_type}_commodity_{self.commodity_value}"

    def get_key(self) -> str:
        """
        Generates a stable, string-based key for this context object.
        Example: "low-new-high"
        """
        return f"{self.season.value}-{self.customer_type.value}-{self.commodity_value.value}"

# --- Helper function to generate all possible context combinations ---
def generate_all_domain_contexts(possible_seasons: list[Season],
                                 possible_customer_types: list[CustomerType],
                                 possible_commodities_values: list[CommodityValue]) -> list[Context]:
    """
    Generates a list of all possible Context objects from the given Enum members.

    Args:
        possible_seasons (list[Season]): A list of all possible Season Enum members (e.g., [Season.LOW, Season.HIGH]).
        possible_customer_types (list[CustomerType]): A list of all possible CustomerType Enum members.
        possible_commodities_values (list[CommodityValue]): A list of all possible Commodity Enum members.

    Returns:
        list[Context]: A list containing all unique Context combinations.
    """
    all_contexts = []
    # itertools.product will correctly iterate over lists of Enum members
    for combination in itertools.product(possible_seasons, possible_customer_types, possible_commodities_values):
        all_contexts.append(Context(season=combination[0],
                                    customer_type=combination[1],
                                    commodity_value=combination[2]))
    return all_contexts

# --- Example Usage ---
if __name__ == '__main__':
    # Using Enum members directly

    """
    context1 = Context(season=Season.HIGH, customer_type=CustomerType.NEW, commodity=Commodity.ELECTRONICS)
    context2 = Context(season=Season.LOW, customer_type=CustomerType.RECURRING, commodity=Commodity.APPAREL)
    context3 = Context(season=Season.HIGH, customer_type=CustomerType.NEW, commodity=Commodity.ELECTRONICS) # Same as context1
    """
    context1 = Context(season=Season.HIGH, customer_type=CustomerType.NEW, commodity_value=CommodityValue.HIGH)
    context2 = Context(season=Season.LOW, customer_type=CustomerType.RECURRING, commodity_value=CommodityValue.LOW)
    context3 = Context(season=Season.HIGH, customer_type=CustomerType.NEW, commodity_value=CommodityValue.HIGH)

    print(f"Context 1: {context1}")
    print(f"Context 2: {context2}")
    print(f"Context 3 (repr): {repr(context3)}")

    # Test equality (Enum members are singletons for a given value or are equal if values match)
    print(f"\nContext 1 equals Context 2? {context1 == context2}") # False
    print(f"Context 1 equals Context 3? {context1 == context3}") # True

    # Test hashing and dictionary usage
    context_data = {}
    context_data[context1] = "Data for context 1"
    context_data[context2] = "Data for context 2"

    print(f"\nData for context1 (using context1 as key): {context_data[context1]}")
    print(f"Data for context1 (using context3 as key): {context_data[context3]}") # context3 is equal to context1

    print(f"\nHash of context1: {hash(context1)}")
    print(f"Hash of context2: {hash(context2)}")
    print(f"Hash of context3: {hash(context3)}") # Should be same as hash(context1)

    # Generate all possible contexts
    # To get all members of an Enum: list(MyEnum)
    all_seasons = list(Season)
    all_customer_types = list(CustomerType)
    all_commodities_values = list(CommodityValue)

    all_possible_contexts_list = generate_all_domain_contexts(all_seasons, all_customer_types, all_commodities_values)
    print(f"\nTotal number of unique context combinations: {len(all_possible_contexts_list)}")
    print("List of all generated contexts (first 5 for brevity):")
    for i, ctx in enumerate(all_possible_contexts_list):
        if i < 5:
            print(f"  {repr(ctx)}") # Use repr for clarity here
        elif i == 5:
            print("  ...")
            break

    # Example of trying to create a Context with an invalid type
    try:
        invalid_context = Context(season="summer_string", customer_type=CustomerType.NEW, commodity=CommodityValue.LOW)
    except TypeError as e:
        print(f"\nError creating invalid context: {e}")

    # Example of accessing Enum members
    print(f"\nAccessing Enum members:")
    print(f"Season.HIGH: {Season.HIGH}")
    print(f"Season.HIGH.name: {Season.HIGH.name}")   # 'HIGH'
    print(f"Season.HIGH.value: {Season.HIGH.value}") # 'high'