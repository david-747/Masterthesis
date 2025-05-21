# MiscShipping.py

from collections import defaultdict # We might not need defaultdict here directly, but it's good for other parts.

class Product:
    def __init__(self, product_id: str, name: str, description: str = "", default_currency: str = "USD"):
        self.product_id = product_id
        self.name = name
        self.description = description
        self.default_currency = default_currency # e.g., the currency this product is usually listed in

    def __repr__(self):
        return f"Product(id='{self.product_id}', name='{self.name}', currency='{self.default_currency}')"

    def __hash__(self):
        return hash(self.product_id)

    def __eq__(self, other):
        if isinstance(other, Product):
            return self.product_id == other.product_id
        return False

class Price:
    """
    Represents a specific price amount with its currency.
    """
    def __init__(self, amount: float, currency: str):
        if not isinstance(amount, (int, float)):
            raise ValueError("Price amount must be a number.")
        if not isinstance(currency, str) or len(currency) != 3: # Basic check for currency code
            raise ValueError("Currency must be a 3-letter string code (e.g., 'USD').")

        self.amount = amount
        self.currency = currency.upper()

    def __repr__(self):
        return f"Price(amount={self.amount:.2f}, currency='{self.currency}')"

    def __str__(self):
        return f"{self.amount:.2f} {self.currency}"

    def __eq__(self, other):
        if isinstance(other, Price):
            return self.amount == other.amount and self.currency == other.currency
        return False

    def __hash__(self):
        return hash((self.amount, self.currency))

class PriceVector:
    def __init__(self, vector_id, name: str = ""):
        self.vector_id = vector_id
        self.name = name
        # Prices: maps Product object (or product_id) to its Price object under this vector
        self.prices_per_product = {} # Key: Product object, Value: Price object

    def set_price(self, product: Product, price_object: Price):
        """Sets the price (as a Price object) for a specific product under this price vector."""
        if not isinstance(product, Product):
            raise ValueError("Invalid product object provided for setting price.")
        if not isinstance(price_object, Price):
            raise ValueError("Invalid Price object provided. Please provide an instance of the Price class.")
        self.prices_per_product[product] = price_object

    def get_price_object(self, product: Product) -> Price:
        """Gets the Price object for a specific product under this price vector."""
        if not isinstance(product, Product):
            raise ValueError("Invalid product object provided for getting price.")
        if product not in self.prices_per_product:
            # Consider how to handle this: raise error, or return None, or a default Price?
            # For now, raising an error as it's likely a setup issue.
            raise KeyError(f"Price for product '{product.name}' (ID: {product.product_id}) "
                           f"not defined in PriceVector '{self.name}' (ID: {self.vector_id}).")
        return self.prices_per_product[product]

    def get_all_price_objects_ordered(self, product_order: list[Product]) -> list[Price]:
        """Returns a list of Price objects in a specific product order."""
        return [self.get_price_object(p) for p in product_order]

    def __repr__(self):
        prices_repr = ", ".join([f"{p.name}: {pr}" for p, pr in self.prices_per_product.items()])
        return (f"PriceVector(id='{self.vector_id}', name='{self.name}', "
                f"prices={{ {prices_repr} }})")

    def __hash__(self):
        return hash(self.vector_id)

    def __eq__(self, other):
        if isinstance(other, PriceVector):
            return self.vector_id == other.vector_id
        return False

# --- Example of how you might define and use these classes ---
if __name__ == '__main__':
    # Define Products
    product_TEU_USD = Product(product_id="P001", name="20ft TEU Slot", default_currency="USD")
    product_FEU_USD = Product(product_id="P002", name="40ft FEU Slot", default_currency="USD")
    product_TEU_EUR = Product(product_id="P003", name="20ft TEU Slot - EU Service", default_currency="EUR")

    all_my_products = [product_TEU_USD, product_FEU_USD, product_TEU_EUR]

    # Define Price Vectors
    pv0_usd = PriceVector(vector_id=0, name="Low Season Base USD")
    pv0_usd.set_price(product_TEU_USD, Price(5500.00, "USD"))
    pv0_usd.set_price(product_FEU_USD, Price(8500.00, "USD"))
    # Let's say this price vector doesn't define a price for product_TEU_EUR
    # pv0_usd.set_price(product_TEU_EUR, Price(5000.00, "EUR")) # Or it could, if mixed currency vectors are allowed

    pv1_usd = PriceVector(vector_id=1, name="Peak Season Surge USD")
    pv1_usd.set_price(product_TEU_USD, Price(6500.00, "USD"))
    pv1_usd.set_price(product_FEU_USD, Price(9800.00, "USD"))

    pv2_eur = PriceVector(vector_id=2, name="EU Promo EUR")
    pv2_eur.set_price(product_TEU_EUR, Price(4800.00, "EUR"))
    # pv2_eur.set_price(product_TEU_USD, Price(5200.00, "EUR")) # A USD product priced in EUR

    all_my_price_vectors = [pv0_usd, pv1_usd, pv2_eur]
    price_indices = [pv.vector_id for pv in all_my_price_vectors]

    print(product_TEU_USD)
    print(Price(123.456, "EUR")) # Test Price __str__ and __repr__

    print(f"\n--- Price Vector 0 ({pv0_usd.name}) ---")
    print(pv0_usd)
    price_obj_teu_pv0 = pv0_usd.get_price_object(product_TEU_USD)
    print(f"Price of {product_TEU_USD.name} under pv0: {price_obj_teu_pv0} (Amount: {price_obj_teu_pv0.amount}, Currency: {price_obj_teu_pv0.currency})")

    try:
        print(pv0_usd.get_price_object(product_TEU_EUR))
    except KeyError as e:
        print(f"Error retrieving price for EU TEU under pv0: {e}")


    print(f"\n--- Price Vector 2 ({pv2_eur.name}) ---")
    print(pv2_eur)
    price_obj_teu_eur_pv2 = pv2_eur.get_price_object(product_TEU_EUR)
    print(f"Price of {product_TEU_EUR.name} under pv2: {price_obj_teu_eur_pv2}")

    # How it might be used in the LP (conceptual)
    # For the LP: objective is sum of p_ik * d_ik
    # p_ik would be: price_vector_k.get_price_object(product_i).amount
    # (assuming d_ik is the demand, and the LP works with numerical values for revenue)
    # You'd need to ensure currency consistency or handle conversions if the LP sums revenues.

    # Example for agent:
    # sampled_theta[context_obj][product_TEU_USD][pv0_usd.vector_id] = sampled_demand_rate
    # actual_price_for_lp = pv0_usd.get_price_object(product_TEU_USD).amount