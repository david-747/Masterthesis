import numpy as np
# Assuming your custom classes are in these files, adjust if necessary
from Context import Context
from MiscShipping import Product
import json


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
                 all_possible_price_indices: list[int],
                 prior_beliefs_path: str = None):
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

        # --- ADD THIS BLOCK TO LOAD PRIOR BELIEFS ---
        if prior_beliefs_path:
            self._load_prior_beliefs(prior_beliefs_path)
            print(f"Successfully loaded prior beliefs from {prior_beliefs_path}")
        # --- END OF NEW BLOCK ---

    def _load_prior_beliefs(self, filepath: str):
        """Loads beliefs from a JSON file and populates the posterior_params."""
        try:
            with open(filepath, 'r') as f:
                beliefs = json.load(f)

            for context_str, product_price_beliefs in beliefs.items():
                # This part is a bit tricky because we need to reconstruct the context key.
                # A simple way is to find the context object that matches the string representation.
                matching_context = next((ctx for ctx in self.all_contexts if str(ctx) == context_str), None)
                if not matching_context:
                    continue  # Skip if the context from the file is not in the current simulation

                context_key = matching_context.get_key()

                for arm_key, params in product_price_beliefs.items():
                    # Arm key is in the format "product_id-price_vector_id"
                    product_id, price_vector_id_str = arm_key.split('-')
                    price_vector_id = int(price_vector_id_str)

                    internal_key = (context_key, product_id, price_vector_id)
                    self.posterior_params[internal_key] = (params['alpha'], params['beta'])

        except FileNotFoundError:
            print(f"Warning: Prior beliefs file not found at {filepath}. Starting with default priors.")
        except Exception as e:
            print(f"An error occurred while loading prior beliefs: {e}")

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

    # In DelayedTSAgent.py, inside the DelayedTSAgent class
    def get_beliefs(self):
        """
        Returns the learned alpha and beta parameters for each context-action pair.
        """
        beliefs = {}
        for context in self.all_contexts:
            context_key_for_storage = context.get_key()
            context_str = str(context)  # For JSON serialization
            beliefs[context_str] = {}

            for product in self.all_products:
                product_id_for_storage = product.product_id

                for pv_index in self.all_price_indices:
                    # Construct the internal key exactly as the agent does
                    internal_key = (context_key_for_storage, product_id_for_storage, pv_index)

                    # --- THE FIX ---
                    # Get the (alpha, beta) tuple from the correct variable: self.posterior_params
                    alpha, beta = self.posterior_params.get(internal_key, (1, 1))
                    # --- END FIX ---

                    # The arm key for the output file
                    arm_key_output = f"{product.product_id}-{pv_index}"
                    prob = alpha / (alpha + beta)

                    beliefs[context_str][arm_key_output] = {
                        'alpha': alpha,
                        'beta': beta,
                        'prob_success': prob
                    }
        return beliefs