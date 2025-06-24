import numpy as np
# Assuming your custom classes are in these files, adjust if necessary
from Context import Context
from MiscShipping import Product


class DelayedTSAgent:
    """
    A Thompson Sampling agent designed to handle delayed feedback.

    This agent maintains a Beta distribution (defined by alpha and beta parameters)
    for each "arm" of the contextual multi-armed bandit. An arm is a unique
    combination of (context, product, price_vector).
    """

    def __init__(self,
                 all_possible_contexts: list[Context],
                 all_possible_products: list[Product],
                 all_possible_price_indices: list[int]):
        """
        Initializes the agent.
        """
        self.all_contexts = all_possible_contexts
        self.all_products = all_possible_products
        self.all_price_indices = all_possible_price_indices

        # This dictionary will store the agent's beliefs.
        # Key: A tuple (context_hash, product_id, price_vector_id)
        # Value: A tuple (alpha, beta)
        self.posterior_params = {}

    def sample_theta_for_each_arm(self) -> dict:
        """
        Performs the Thompson Sampling step.

        This version creates a NESTED dictionary to match the format expected
        by the LP Solver: {context: {product: {price_id: probability}}}
        """
        sampled_theta_nested = {}

        for context in self.all_contexts:
            # Use the actual Context object as the first-level key
            sampled_theta_nested[context] = {}
            for product in self.all_products:
                # Use the actual Product object as the second-level key
                sampled_theta_nested[context][product] = {}

        # Now that the nested structure is initialized, fill it with samples.
        for context in self.all_contexts:
            context_key_for_storage = context.get_key()  # This is our stable string key for internal storage

            for product in self.all_products:
                product_id_for_storage = product.product_id

                for pv_index in self.all_price_indices:
                    # Use the consistent tuple key to get our internal belief
                    internal_key = (context_key_for_storage, product_id_for_storage, pv_index)
                    alpha, beta = self.posterior_params.get(internal_key, (1, 1))

                    # Sample the belief
                    sample = np.random.beta(alpha, beta)

                    # Populate the nested dictionary using the OBJECTS as keys
                    sampled_theta_nested[context][product][pv_index] = sample

        return sampled_theta_nested

    def update_posterior(self, context: Context, product: Product, price_vector_id: int, success: bool):
        """
        Updates the posterior distribution for a given arm based on feedback.
        This is the core learning mechanism.
        """
        # Construct the key in the EXACT SAME WAY as in the sampling method
        # to ensure consistency.
        context_key = context.get_key()  # Use the stable key
        key = (context_key, product.product_id, price_vector_id)

        # Retrieve the current parameters for this key, or the default prior (1, 1).
        alpha, beta = self.posterior_params.get(key, (1, 1))

        # Update the parameters based on the feedback outcome.
        if success:
            alpha += 1  # Increment alpha for a success (sale)
        else:
            beta += 1  # Increment beta for a failure (no sale)

        # Store the new parameters back into our beliefs dictionary.
        self.posterior_params[key] = (alpha, beta)