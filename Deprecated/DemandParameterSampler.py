import numpy as np
from collections import defaultdict


class DemandParameterSampler:
    def __init__(self, products, price_indices, contexts, demand_model_type='bernoulli'):
        self.products = products
        self.price_indices = price_indices
        self.contexts = contexts
        self.demand_model_type = demand_model_type

        # posterior_params will store {'alpha': count_successes + prior_alpha, 'beta': count_failures + prior_beta}
        # Initialize with Beta(1,1) prior for each (context, product, price_idx)
        self.posterior_params = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'alpha': 1, 'beta': 1})))

    def sample_theta_for_period_t(self):
        """
        Samples a complete set of mean demand rates theta(t) from the current posterior.
        theta(t) = {context: {product_id: {price_idx: sampled_mean_demand}}}
        """
        sampled_theta_t = defaultdict(lambda: defaultdict(dict))
        if self.demand_model_type == 'bernoulli':
            for ctx in self.contexts:
                for prod_id in self.products:
                    for p_idx in self.price_indices:
                        params = self.posterior_params[ctx][prod_id][p_idx]
                        sampled_mean_demand = np.random.beta(params['alpha'], params['beta'])
                        sampled_theta_t[ctx][prod_id][p_idx] = sampled_mean_demand
        # Add other demand models (e.g., Poisson with Gamma posterior) if needed
        return sampled_theta_t

    def update_posterior(self, observed_context, product_id, price_idx, num_successes, num_trials=1):
        """
        Updates the posterior distribution for a given context, product, and price index.
        For Bernoulli, num_successes is 0 or 1 if num_trials is 1.
        """
        if self.demand_model_type == 'bernoulli':
            # Assuming num_trials is 1 for a single observation per period
            is_sale = num_successes > 0

            # Ensure the entry exists if it was never used (though init should cover it)
            current_params = self.posterior_params[observed_context][product_id][price_idx]

            if is_sale:
                current_params['alpha'] += 1
            else:
                current_params['beta'] += 1
        # Add other demand models