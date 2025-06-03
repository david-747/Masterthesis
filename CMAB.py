from collections import defaultdict
import numpy as np


# It's good practice to use type hints, including for classes defined elsewhere.
# You can use forward references (strings) if the classes are in files that
# might cause circular imports, or just import them directly if structure allows.
# from DelayedTSAgent import DelayedTSAgent
# from MiscShipping import Product, PriceVector, Price
# from Context import DomainContext

# CMAB is the orchestrator of the model
class CMAB:
    #takes instance of DelayedTSAgent as one of its main components
    #takes lp_solver_function, function (defined elsewhere) used to solve a Linera Programming problem
    #LP helps decide the optimal probabilities of offering different price vectors considering resource constraints
    # __init__ store information about products, price vectors, resource consumption by products, initial ressource
    #inventory and total time periods
    def __init__(self,
                 agent: 'DelayedTSAgent',
                 lp_solver_function,
                 all_products: list['Product'],
                 all_price_vectors_map: dict[int, 'PriceVector'],  # e.g., {pv_id: PriceVector_object}
                 resource_consumption_matrix: np.ndarray,  # Shape: (num_products, num_resources)
                 initial_resource_inventory: np.ndarray,  # Shape: (num_resources,)
                 total_time_periods: int,
                 context_probabilities: dict['DomainContext', float] = None):
        """
        Orchestrates a Contextual Multi-Armed Bandit strategy with delayed feedback,
        structurally following Ferreira et al. (2018) Algorithm 4,
        but adapted for delayed feedback using a DelayedTSAgent.

        Args:
            agent: An instance of DelayedTSAgent.
            lp_solver_function: A function to solve the LP (Step 2 of Alg 4).
                Expected signature: solve_lp(
                                        sampled_theta_t, # From agent.get_sampled_theta()
                                        resource_constraints_c_j,
                                        all_contexts, # From agent.all_contexts
                                        all_products, # List of Product objects
                                        all_price_indices, # List of price vector IDs
                                        all_price_vectors_map, # To get p_ik (actual prices)
                                        resource_consumption_matrix_A_ij,
                                        context_probabilities # Optional P(ksi) for E_ksi in LP objective
                                    ) -> dict: {context_obj: {price_vector_id: probability_x_ksi_k}}
            all_products: List of Product objects.
            all_price_vectors_map: Dictionary mapping price_vector_id to PriceVector objects.
            resource_consumption_matrix: Numpy array (num_products x num_resources) for a_ij.
            initial_resource_inventory: Numpy array (num_resources,) for initial I_j.
            total_time_periods: Total periods T, used for calculating c_j = I_j / T.
            context_probabilities: Optional dict mapping DomainContext to its probability P(ksi).
                                     Used if the LP explicitly calculates E_ksi.
        """
        self.agent = agent
        self.lp_solver = lp_solver_function

        self.all_products = all_products
        # Create a mapping from product_id to product_idx for the resource_consumption_matrix
        self.product_to_idx_map = {product.product_id: i for i, product in enumerate(all_products)}

        self.all_price_vectors_map = all_price_vectors_map
        self.all_price_indices = sorted(list(all_price_vectors_map.keys()))  # Consistent with agent

        self.resource_consumption_matrix_A_ij = resource_consumption_matrix
        self.initial_resource_inventory_I_j = initial_resource_inventory
        self.total_time_periods_T = total_time_periods
        # c_j = I_j / T as per Ferreira et al. (2018) (Section 2.2, used in LP)
        self.resource_constraints_c_j = self.initial_resource_inventory_I_j / self.total_time_periods_T

        self.context_probabilities = context_probabilities
        if self.context_probabilities:
            # Basic validation for context_probabilities
            if not all(isinstance(ctx, type(next(iter(self.agent.all_contexts)))) for ctx in
                       self.context_probabilities.keys()):
                raise ValueError("Keys in context_probabilities must be of the same type as agent's contexts.")
            if abs(sum(self.context_probabilities.values()) - 1.0) > 1e-6:
                raise ValueError("Context probabilities P(ksi) must sum to 1.0.")

        self.current_lp_solution_x_ksi_k = None  # Stores the result of LP optimization (Step 2)


    #at start of each period t, the simulator environment will call this method with any feedback that has arrived
    #CMAB simply passes this feedback along to self.agent.process_arrived_feedback(feedback_id, success)
    #this updates the agents demand model before making new decisions for the c
    def process_feedback_for_agent(self, feedback_to_process: list[tuple[any, bool]]):
        """
        Passes arrived feedback to the agent for posterior updates.
        This should be called at the START of each time period t by the simulator.

        Args:
            feedback_to_process: A list of tuples, where each tuple is
                                 (product_specific_feedback_id, observed_success_bool).
                                 The product_specific_feedback_id is what was generated by
                                 `select_action_and_record_for_feedback`.
        """
        for feedback_id, success in feedback_to_process:
            self.agent.process_arrived_feedback(feedback_id, success)


    #performs Step 1 and 2 of Algorithm 4 of Ferreira et al
    #Step 1 (Sample Demand): calls self.agent.get_sampled_theta() to get get the agents current best guess of demand
    #                        rate for all context-product-price combinations
    #Step 2 (Optimize Prices): calls self.lp_solver() function. Solver uses sampled demand rates, actual prices,
    #                          resource consumption rates and ressource constraints to determine an optimal
    #                          randomized pricing policy
    #                          this policy is stored in self.current_lp_solution_x_ksi_k and looks like:
    #                          {context_object: {price_vector_id: probability_of_offering_this_price_vector}}
    def determine_pricing_policy_for_period(self):
        """
        Performs Steps 1 and 2 of Algorithm 4 (delay-adapted):
        1. Sample theta(t) from the agent's current posteriors.
        2. Solve the LP to get the optimal x_ksi_k probabilities.
        This method updates `self.current_lp_solution_x_ksi_k`.
        """
        # Step 1: Sample Demand theta(t)
        # The agent's posteriors should have been updated by `process_feedback_for_agent`
        sampled_theta_t = self.agent.get_sampled_theta()
        # Expected structure: {context_obj: {product_obj: {price_vector_id: sampled_mean_demand}}}

        # Step 2: Optimize Prices Given Sampled Demand (Solve LP)
        # The LP solver will use sampled_theta_t to get d_ik(ksi|theta(t)).
        # It will use all_price_vectors_map to get p_ik (actual price amounts).
        # It will use resource_consumption_matrix for a_ij.
        # It will use self.resource_constraints_c_j for c_j.
        self.current_lp_solution_x_ksi_k = self.lp_solver(
            sampled_theta_t=sampled_theta_t,
            resource_constraints_c_j=self.resource_constraints_c_j,
            all_contexts=self.agent.all_contexts,
            all_products=self.all_products,
            all_price_indices=self.agent.all_price_indices,
            all_price_vectors_map=self.all_price_vectors_map,
            resource_consumption_matrix_A_ij=self.resource_consumption_matrix_A_ij,
            context_probabilities=self.context_probabilities,
            product_to_idx_map=self.product_to_idx_map  # For indexing into A_ij
        )
        # Expected structure of self.current_lp_solution_x_ksi_k:
        # {context_obj: {price_vector_id: probability_x_ksi_k}}

    #performs Step 3 of algorithm
    #Step 3: 3.1 takes observed_realized_context for the current period
    #        3.2 looks up the pricing policy (probabilities) for this context from self.current_lp_solution_x_ksi_k
    #        3.3 randomly selects a chosen_price_vector_id based on these probabilities
    #            might also result in choosing p_infinity (represented as None), meaning no products are offered
    #        3.4 if specific price vector is chosen:
    #            for each product, it generates a unique feedback_id (touple of time, context, price vector id,
    #            product id) and calls self.agent.record_action_taken() with this feedback id and other details
    #            this tells the agent to expect feedback for this specfic action on this product later
    #            returns the chosen price vector ID and map of {Product_obj: feedback_id}
    #        3.5 if p_infinity is chosen: it returns None, None
    def select_action_and_record_for_feedback(self,
                                              observed_realized_context: 'DomainContext',
                                              current_time_t: int) -> tuple[int | None, dict['Product', any] | None]:
        """
        Performs Step 3 of Algorithm 4 (Offer Price) and records the action with the agent
        for future feedback processing (part of delay-adapted Step 4).

        Args:
            observed_realized_context: The actual DomainContext object observed in this period.
            current_time_t: The current time period index, t.

        Returns:
            A tuple:
            - chosen_price_vector_id (int | None): ID of the PriceVector chosen, or None for p_infinity.
            - product_specific_feedback_ids_map (dict | None):
                If a price vector is chosen, maps {Product_obj: product_specific_feedback_id}.
                This ID is what the simulator uses to report feedback for that product from that action.
                Returns None if p_infinity is chosen.
        """
        if self.current_lp_solution_x_ksi_k is None:
            raise RuntimeError("LP solution (x_ksi_k) not available. "
                               "Call determine_pricing_policy_for_period() first.")

        if observed_realized_context not in self.current_lp_solution_x_ksi_k:
            # This can happen if the LP solution doesn't cover all contexts,
            # or if an unexpected context appears.
            print(f"Warning: Observed context {observed_realized_context} not found in current LP solution. "
                  f"Offering p_infinity.")
            chosen_price_vector_id = None
        else:
            probabilities_for_context = self.current_lp_solution_x_ksi_k[observed_realized_context]
            # Ensure all possible price indices are considered for p_infinity calculation
            # even if some have zero probability from the LP.

            prob_sum_for_specific_prices = 0.0
            choices_for_sampling = []
            probs_for_sampling = []

            for p_idx in self.agent.all_price_indices:
                prob = probabilities_for_context.get(p_idx, 0.0)  # Default to 0 if not in LP solution for this context
                if prob > 1e-9:  # Only consider if probability is non-negligible
                    choices_for_sampling.append(p_idx)
                    probs_for_sampling.append(prob)
                prob_sum_for_specific_prices += prob

            if not (0 <= prob_sum_for_specific_prices <= 1.00001):
                print(f"Warning: Sum of probabilities for price vectors in context "
                      f"{observed_realized_context} is {prob_sum_for_specific_prices}. Check LP constraint sum(x_ksi_k) <= 1.")
                prob_sum_for_specific_prices = min(1.0, max(0.0, prob_sum_for_specific_prices))  # Clamp

            prob_p_infinity = 1.0 - prob_sum_for_specific_prices

            if prob_p_infinity > 1e-9:  # If there's a chance for p_infinity
                choices_for_sampling.append(None)  # None represents p_infinity
                probs_for_sampling.append(prob_p_infinity)

            if not choices_for_sampling:  # Should not happen if p_infinity has prob > 0
                print(f"Warning: No valid choices for context {observed_realized_context}. Offering p_infinity.")
                chosen_price_vector_id = None
            else:
                # Normalize probabilities for np.random.choice
                normalized_probs = np.array(probs_for_sampling) / sum(probs_for_sampling)
                chosen_price_vector_id = np.random.choice(choices_for_sampling, p=normalized_probs)

        # --- Record action details with the agent for each product if a price vector was chosen ---
        product_specific_feedback_ids_map = {}
        if chosen_price_vector_id is not None:
            for product_obj in self.all_products:
                # This unique ID will be used by the simulator to report feedback for this specific product
                # resulting from this specific action (t, context, price_vector_id).
                feedback_id = (
                current_time_t, observed_realized_context, chosen_price_vector_id, product_obj.product_id)
                self.agent.record_action_taken(
                    time_t=current_time_t,
                    context_obj=observed_realized_context,
                    product_obj=product_obj,
                    price_idx=chosen_price_vector_id,
                    action_id=feedback_id
                )
                product_specific_feedback_ids_map[product_obj] = feedback_id
            return chosen_price_vector_id, product_specific_feedback_ids_map
        else:  # p_infinity chosen
            return None, None

    # Step 4 (Update Estimate of Parameter) is now handled by:
    # 1. `process_feedback_for_agent()` being called by the simulator at the start of a period.
    # 2. `select_action_and_record_for_feedback()` calling `agent.record_action_taken()`.