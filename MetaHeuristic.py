# MetaHeuristic.py (or EpisodicPlanner.py)

import numpy as np
from collections import defaultdict

# Attempt to import your custom classes for type hinting.
# These will not cause an error if the files are not found during standalone execution of this sketch,
# as they are used as string hints or within comments.
try:
    from DelayedTSAgent import DelayedTSAgent
    from MiscShipping import Product, PriceVector, Price
    from Context import DomainContext
except ImportError:
    # Define dummy classes if imports fail, for type hinting purposes in a standalone context
    class DelayedTSAgent:
        pass


    class Product:
        pass


    class PriceVector:
        pass


    class Price:
        pass


    class DomainContext:
        pass


class EpisodicPlanner:
    def __init__(self,
                 all_products: list['Product'],
                 all_price_vectors_map: dict[int, 'PriceVector'],
                 initial_total_capacity_per_product: dict['Product', int],
                 episode_horizon_H: int,
                 max_corrupted_episodes_q_bar: int,
                 num_total_episodes_K_for_bonus: int,
                 episodic_lp_solver_function,  # MOVED: Non-default argument
                 # Default arguments now follow:
                 delta_bonus: float = 0.1,
                 s_cardinality_estimate: int = 1000,
                 demand_model_provider: 'DelayedTSAgent' = None
                 ):
        """
        Planner inspired by BEEP-LP from Hu et al. (2024).
        """
        self.all_products = all_products
        self.product_to_idx_map = {p: i for i, p in enumerate(all_products)}
        self.idx_to_product_map = {i: p for i, p in enumerate(all_products)}
        self.num_products_D = len(all_products)  # D from Hu et al.

        self.all_price_vectors_map = all_price_vectors_map  # {pv_id: PriceVector_object}
        self.mdp_actions = sorted(list(all_price_vectors_map.keys()))  # Actions are price_vector_ids
        self.A_cardinality = len(self.mdp_actions)  # |A| for bonus

        # ksi_i: capacity for each product_type (D-dimensional)
        self.episode_capacity_ksi = np.array(
            [initial_total_capacity_per_product.get(p, 0.0) for p in all_products]
        )

        self.episode_horizon_H = episode_horizon_H  # H

        # For bonus calculation N_k(s,a) (Hu et al. Eq. 7)
        # MDP State 's': (time_step_h, inventory_tuple_for_all_products)
        # MDP Action 'a': price_vector_id
        self.state_action_visitation_counts_N = defaultdict(lambda: defaultdict(lambda: 1))  # Init to 1

        self.max_corrupted_episodes_q_bar = max_corrupted_episodes_q_bar  # q_bar
        self.num_total_episodes_K_for_bonus = num_total_episodes_K_for_bonus  # K in bonus formula
        self.delta_bonus = delta_bonus
        self.S_cardinality_estimate = s_cardinality_estimate  # |S| for bonus

        self.episodic_lp_solver = episodic_lp_solver_function
        self.current_episode_policy = None  # pi_k: {mdp_state: {action_pv_id: probability}}

        self.learned_demand_model = demand_model_provider  # Expected structure from DelayedTSAgent

    def update_learned_demand_model(self, demand_model_from_bandit: DelayedTSAgent):
        """
        Receives the latest demand model from the contextual bandit (DelayedTSAgent).
        The 'demand_model_from_bandit' should allow fetching expected demand rates.
        """
        self.learned_demand_model = demand_model_from_bandit
        # print("EpisodicPlanner: Updated learned demand model.")

    def _get_mdp_state_tuple(self, time_step_h: int, current_inventories_list: list[float]) -> tuple:
        """ Helper to create a consistent, hashable state representation for the MDP. """
        if len(current_inventories_list) != self.num_products_D:
            raise ValueError(f"Inventory list length {len(current_inventories_list)} "
                             f"does not match num_products {self.num_products_D}")
        return (time_step_h,) + tuple(current_inventories_list)

    def _calculate_bonus_b(self, mdp_state: tuple, action_pv_id: int, current_episode_k_for_bonus: int) -> float:
        """ Calculates bonus b_k(s,a) as per Hu et al. (2024) Eq. (4). """
        N_s_a = self.state_action_visitation_counts_N[mdp_state].get(action_pv_id, 1)  # Default to 1 if not visited

        # Using K from the bonus formula as specified by Hu et al.
        # This K is the total number of episodes the system is expected to run for, used in the theoretical bound.
        # current_episode_k_for_bonus is the current k for N_k(s,a)
        log_term_numerator = (64 * self.S_cardinality_estimate * self.A_cardinality *
                              self.episode_horizon_H * (self.num_products_D + 1) *
                              (self.num_total_episodes_K_for_bonus ** 2))

        if log_term_numerator <= 0 or self.delta_bonus <= 0 or N_s_a <= 0:  # Prevent math errors
            return 0.0

        try:
            term1_sqrt_val = np.log(log_term_numerator / self.delta_bonus)
            if term1_sqrt_val < 0: term1_sqrt_val = 0  # log of num < 1 can be negative

            bonus = self.episode_horizon_H * np.sqrt(2 * term1_sqrt_val / N_s_a) + \
                    (self.max_corrupted_episodes_q_bar * self.episode_horizon_H ** 2) / N_s_a
        except (OverflowError, ValueError):
            bonus = 2 * self.episode_horizon_H  # Fallback in case of math error

        return min(2 * self.episode_horizon_H, bonus)

    def _get_expected_outcomes(self, price_vector_id: int, current_global_context: DomainContext) -> tuple[
        float, np.ndarray]:
        """
        Uses the learned_demand_model to get estimated mean reward (r_hat) and
        mean consumptions (c_hat_i) for a given price vector in a context.
        This does NOT include the MDP state 's' directly, assumes context is primary driver for demand.
        """
        if self.learned_demand_model is None:
            # print("Warning: Learned demand model not available for getting expected outcomes.")
            return 0.0, np.zeros(self.num_products_D)

        price_vector_obj = self.all_price_vectors_map.get(price_vector_id)
        if not price_vector_obj:
            return 0.0, np.full(self.num_products_D, float('inf'))  # Penalize

        total_expected_reward = 0.0
        expected_consumptions = np.zeros(self.num_products_D)

        for i, product_obj in enumerate(self.all_products):
            try:
                # Fetch posterior for the specific product, price_vector, and context
                posterior = self.learned_demand_model.posterior_params[current_global_context][product_obj][
                    price_vector_id]
                # Expected demand = alpha / (alpha + beta) for Beta distribution
                mean_demand = posterior['alpha'] / (posterior['alpha'] + posterior['beta'])

                price_details = price_vector_obj.get_price_object(product_obj)  # Price object
                # Assuming LP needs revenue in a consistent currency, or handles conversion
                total_expected_reward += mean_demand * price_details.amount
                expected_consumptions[i] = mean_demand  # Assuming 1 unit sold = 1 unit of product i capacity consumed
            except KeyError:
                # This combination might not have been learned by the bandit or price not set
                # print(f"Warning: No demand/price info for {product_obj}, pv_id {price_vector_id}, context {current_global_context}")
                pass  # Defaults to 0 for this product's reward/consumption

        return total_expected_reward, expected_consumptions

    def plan_policy_for_new_episode(self, current_episode_k: int, current_global_context: DomainContext):
        """
        Plans the policy pi_k for the upcoming episode k by preparing inputs for and calling the episodic_lp_solver.
        Corresponds to solving Eq. 5 from Hu et al. (2024).
        """
        if self.learned_demand_model is None:
            raise RuntimeError("Learned demand model needs to be updated via update_learned_demand_model() first.")

        # The LP solver will need to iterate over all MDP states (s) and actions (a)
        # and for each, it will need r_k(s,a) and c_ik(s,a).
        # r_k(s,a) = hat_r_k(s,a) + b_k(s,a)
        # c_ik(s,a) = hat_c_ik(s,a) - b_k(s,a)
        # hat_r and hat_c come from _get_expected_outcomes based on current_global_context.
        # The bonus b_k(s,a) depends on the MDP state s.

        # The lp_solver_function should be designed to accept these components or
        # functions to compute them.

        # This data structure would be built up to pass to the LP solver.
        # It would contain, for each (state, action) pair:
        # - bonus_enhanced_reward
        # - bonus_adjusted_consumptions
        # - estimated_next_state_distribution (p_hat(s'|s,a)) -> this is the hardest part.
        # For BEEP-LP, the LP is often formulated over occupation measures q_h(s,a).

        # print(f"EpisodicPlanner: Planning policy for episode {current_episode_k}...")
        self.current_episode_policy = self.episodic_lp_solver(
            episode_horizon_H=self.episode_horizon_H,
            # mdp_states_definition: How to define/iterate all MDP states for the LP.
            # For now, this is abstract. Could be a generator or a list.
            all_mdp_actions=self.mdp_actions,
            episode_capacity_ksi=self.episode_capacity_ksi,
            current_episode_k_for_bonus=current_episode_k,  # For bonus calculation timing
            # Pass functions that the LP solver can call to get parameters for any (s,a)
            get_expected_outcomes_func=lambda action_pv_id: self._get_expected_outcomes(action_pv_id,
                                                                                        current_global_context),
            calculate_bonus_func=lambda mdp_state, action_pv_id: self._calculate_bonus_b(mdp_state, action_pv_id,
                                                                                         current_episode_k),
            # The LP solver also needs a way to model state transitions p_hat(s'|s,a)
            # This would also use _get_expected_outcomes to find sales, then update inventory.
            # For a sketch, this is a major part of the LP solver's internal logic.
            # For example, state s = (h, inv_tuple), action a = price_vector_id
            #   sales = _get_expected_outcomes(a, context).consumptions
            #   next_inv = inv_tuple - sales
            #   s_prime = (h+1, next_inv)
            #   p_hat(s_prime | s, a) = 1 (if using expected sales for deterministic transition for sketch)
            #   In reality, sales are stochastic, so p_hat is a distribution.
            num_products=self.num_products_D
        )
        # self.current_episode_policy should be: {mdp_state_tuple: chosen_action_pv_id}
        # or {mdp_state_tuple: {action_pv_id: probability}} if policy is stochastic
        # print(f"EpisodicPlanner: Policy for episode {current_episode_k} planned.")
        return self.current_episode_policy

    def get_action(self, current_time_step_h: int, current_inventories_list: list[float]) -> int:
        """
        Get the action for the current MDP state based on the planned episode policy.
        Also updates the state-action visitation count.
        """
        if self.current_episode_policy is None:
            # print("Warning: Episode policy not planned. Choosing a default/random action.")
            # Fallback: if no policy, maybe choose a default action or random
            chosen_action = np.random.choice(self.mdp_actions)
            temp_mdp_state = self._get_mdp_state_tuple(current_time_step_h, current_inventories_list)
            self.state_action_visitation_counts_N[temp_mdp_state][chosen_action] += 1
            return chosen_action

        current_mdp_state = self._get_mdp_state_tuple(current_time_step_h, current_inventories_list)

        # Assuming policy is {state: action_id} for simplicity in sketch for BEEP-LP greedy policy
        action_to_take = self.current_episode_policy.get(current_mdp_state)

        if action_to_take is None:
            # Policy didn't cover this state (should ideally not happen with a full LP solution)
            # print(f"Warning: State {current_mdp_state} not in policy. Choosing random action.")
            action_to_take = np.random.choice(self.mdp_actions)

        # Record visit for N_k+1(s,a) calculation
        self.state_action_visitation_counts_N[current_mdp_state][action_to_take] += 1
        return action_to_take