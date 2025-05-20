
from collections import defaultdict
import numpy as np


class DelayedTSAgent:
    def __init__(self, all_possible_contexts, all_possible_products, all_possible_price_indices, demand_model_type='bernoulli'):
        self.all_contexts = all_possible_contexts
        self.all_products = all_possible_products
        self.all_price_indices = all_possible_price_indices # e.g., list(range(K))
        self.demand_model_type = demand_model_type

        # This correctly initializes posteriors to priors on first access
        self.posterior_params = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'alpha': 1, 'beta': 1})))

        # --- New for delays ---
        # Stores information about actions for which feedback is pending
        # Key: unique_action_id (e.g., (time_taken, context_hash, action_hash))
        # Value: {'product_id': ..., 'price_idx': ..., 'context': ...}
        self.pending_actions_info = {}

    def get_sampled_theta(self):
        """
        Samples a complete set of mean demand rates theta from the *current* posterior
        for ALL defined combinations of context, product, and price_index.
        """
        # The defaultdict for sampled_theta is for convenient construction of the output dict
        sampled_theta = defaultdict(lambda: defaultdict(dict))

        if self.demand_model_type == 'bernoulli':
            for ctx_key in self.all_contexts:  # Iterate over ALL defined contexts
                for prod_id in self.all_products:  # Iterate over ALL defined products
                    # sampled_theta[ctx_key][prod_id] will be a dict to store p_idx -> sampled_demand
                    for p_idx in self.all_price_indices:  # Iterate over ALL defined price indices

                        # Accessing self.posterior_params here:
                        # If this (ctx_key, prod_id, p_idx) combination has never been
                        # accessed/updated before, the defaultdict structure will ensure
                        # params becomes {'alpha': 1, 'beta': 1} (the prior).
                        # If it has been updated, params will hold the updated alpha/beta.
                        params = self.posterior_params[ctx_key][prod_id][p_idx]

                        sampled_mean_demand = np.random.beta(params['alpha'], params['beta'])

                        # Store the sampled demand for this specific combination
                        sampled_theta[ctx_key][prod_id][p_idx] = sampled_mean_demand

        # Add elif blocks for other demand_model_types if necessary

        return sampled_theta

    def record_action_taken(self, time_t, context, product_id, price_idx, action_id):
        """
        Records that an action was taken, so we know its details when feedback arrives.
        action_id could be simply time_t if only one action per time_t, or more complex.
        """
        # Store details needed to update the correct posterior when feedback arrives
        self.pending_actions_info[action_id] = {
            'time_taken': time_t,
            'context': context, # Store the actual context object or its key
            'product_id': product_id,
            'price_idx': price_idx
        }

    def process_arrived_feedback(self, action_id, observed_success, observed_trials=1):
        """
        Called when feedback for a previously taken action arrives.
        Updates the posterior for the specific (context, product, price) of that action.
        """
        if action_id not in self.pending_actions_info:
            print(f"Warning: Received feedback for unknown action_id {action_id}")
            return

        action_details = self.pending_actions_info.pop(action_id) # Remove from pending
        ctx_key = action_details['context'] # Or how you key contexts
        prod_id = action_details['product_id']
        p_idx = action_details['price_idx']

        #if only binary feedback -> alpha/beta + 1 is sufficient
        if self.demand_model_type == 'bernoulli':
            is_sale = observed_success > 0
            current_params = self.posterior_params[ctx_key][prod_id][p_idx]
            if is_sale:
                current_params['alpha'] += 1
            else:
                current_params['beta'] += 1

        #in case of having multiple observing with arriving feeback
        '''
        else:

            current_params = self.posterior_params[ctx_key][prod_id][p_idx]

            current_params['alpha'] += observed_success
            current_params['beta'] += observed_trials - observed_success
'''