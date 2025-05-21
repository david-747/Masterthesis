# Context.py

import itertools

class Context:
    """
    Represents a multi-domain context for pricing decisions.
    It's designed to be hashable and comparable so it can be used as a dictionary key.
    """
    def __init__(self, season: str, customer_type: str, commodity: str):
        """
        Initializes a context with specific values for each domain.

        Args:
            season (str): The season (e.g., 'low_season', 'high_season').
            customer_type (str): The type of customer (e.g., 'new', 'recurring').
            commodity (str): The commodity type (e.g., 'electronics', 'apparel').
        """
        self.season = season
        self.customer_type = customer_type
        self.commodity = commodity

        # Pre-compute the tuple for hashing and equality, ensuring order
        self._key = (self.season, self.customer_type, self.commodity)

    def __hash__(self):
        """
        Returns a hash value for the context, allowing it to be used in sets and as dict keys.
        """
        return hash(self._key)

    def __eq__(self, other):
        """
        Checks if this context is equal to another context.
        Two contexts are equal if all their domain values are the same.
        """
        if isinstance(other, Context):
            return self._key == other._key
        return False

    def __repr__(self):
        """
        Returns an unambiguous string representation of the context object.
        """
        return (f"DomainContext(season='{self.season}', "
                f"customer_type='{self.customer_type}', "
                f"commodity='{self.commodity}')")

    def __str__(self):
        """
        Returns a user-friendly string representation of the context.
        """
        return f"Context(Season: {self.season}, Customer: {self.customer_type}, Commodity: {self.commodity})"

# --- Optional: Helper function to generate all possible context combinations ---
def generate_all_domain_contexts(possible_seasons, possible_customer_types, possible_commodities):
    """
    Generates a list of all possible DomainContext objects from the given domain values.

    Args:
        possible_seasons (list[str]): A list of all possible season values.
        possible_customer_types (list[str]): A list of all possible customer type values.
        possible_commodities (list[str]): A list of all possible commodity values.

    Returns:
        list[Context]: A list containing all unique DomainContext combinations.
    """
    all_contexts = []
    for combination in itertools.product(possible_seasons, possible_customer_types, possible_commodities):
        all_contexts.append(Context(season=combination[0],
                                          customer_type=combination[1],
                                          commodity=combination[2]))
    return all_contexts

# --- Example Usage (you can put this in a test script or your main simulator setup) ---
if __name__ == '__main__':
    # Define the possible values for each domain
    seasons = ['low', 'mid', 'high']
    customer_types = ['new', 'recurring']
    commodities = ['electronics', 'apparel']

    # Create specific context instances
    context1 = Context(season='high', customer_type='new', commodity='electronics')
    context2 = Context(season='low', customer_type='recurring', commodity='apparel')
    context3 = Context(season='high', customer_type='new', commodity='electronics') # Same as context1

    print(f"Context 1: {context1}")
    print(f"Context 2: {context2}")
    print(f"Context 3 (repr): {repr(context3)}")

    # Test equality
    print(f"\nContext 1 equals Context 2? {context1 == context2}") # False
    print(f"Context 1 equals Context 3? {context1 == context3}") # True

    # Test hashing and dictionary usage
    context_data = {}
    context_data[context1] = "Data for context 1"
    context_data[context2] = "Data for context 2"

    print(f"\nData for context1 (using context1 as key): {context_data[context1]}")
    # context3 is equivalent to context1, so it should retrieve the same data
    print(f"Data for context1 (using context3 as key): {context_data[context3]}")

    print(f"\nHash of context1: {hash(context1)}")
    print(f"Hash of context2: {hash(context2)}")
    print(f"Hash of context3: {hash(context3)}") # Should be same as hash(context1)

    # Generate all possible contexts
    all_possible_contexts_list = generate_all_domain_contexts(seasons, customer_types, commodities)
    print(f"\nTotal number of unique context combinations: {len(all_possible_contexts_list)}")
    print("List of all generated contexts:")
    for ctx in all_possible_contexts_list:
        print(ctx)

    # Now, in your Agent class, you would pass this list:
    # agent = YourAgent(all_possible_contexts=all_possible_contexts_list, ...)
    # And when iterating:
    # for ctx_object in self.all_contexts:
    #     params = self.posterior_params[ctx_object]...