import sys
print("Python executable:", sys.executable)

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.optimize import linprog
from collections import defaultdict, deque
import random

# --- Configuration ---
T = 21*24  # Total time periods
N = 2     # Number of products
M = 3     # Number of resources
K = 5     # Number of price vectors (excluding p_infinity)

# Define price vectors (example: N=2 products)
# p_k = (p_1k, p_2k)
price_vectors = {
    0: np.array([5500, 5900]),
    1: np.array([5600, 6000]),
    2: np.array([5000, 8000]),
    3: np.array([5020, 6200]),
    4: np.array([5990, 6100]),
}
price_indices = list(price_vectors.keys()) # [0, 1, 2, 3, 4]

# Resource consumption: a_ij = units of resource j per unit of product i
# Example: M=3 resources
A = np.array([
    [1, 3, 0],  # Product 0 consumes 1 of res 0, 3 of res 1, 0 of res 2
    [1, 1, 5],  # Product 1 consumes 1 of res 0, 1 of res 1, 5 of res 2
]) # Shape (N, M)

# Initial Inventory
I_initial = np.array([150, 180, 250]) # Example initial inventory for M=3 resources

# Contexts (example: discrete contexts like 'Weekday', 'Weekend')
contexts = ['Weekday', 'Weekend']
context_probs = {'Weekday': 0.7, 'Weekend': 0.3} # Example distribution P(xi)

# --- True Demand Parameters (Unknown to the algorithm) ---
# d*_ikx = true probability of purchase for product i, price k, context x
# Needs to be defined based on the specific problem instance
true_demand_params = {}
for k in price_indices:
    for i in range(N):
        for xi in contexts:
            # Example: Demand decreases with price, slightly higher on weekends
            base_demand = 0.6 - 0.02 * price_vectors[k][i]
            context_multiplier = 1.1 if xi == 'Weekend' else 1.0
            true_demand_params[(i, k, xi)] = np.clip(base_demand * context_multiplier, 0.05, 0.95)

# --- Helper Functions ---
def simulate_true_demand(product_idx, price_vector_idx, context, true_params):
    """ Simulates Bernoulli demand based on true parameters. """
    prob = true_params.get((product_idx, price_vector_idx, context), 0)
    return 1 if random.random() < prob else 0

def simulate_delay(t):
    """ Simulates the delay for feedback from period t. """
    # Example: Poisson delay with mean 5
    # Replace with a more realistic model if needed (e.g., from Blanchet)
    return np.random.poisson(5)
    # Example: Fixed delay
    # return 3

# --- Core Algorithm Components ---

class InventoryManager:
    def __init__(self, initial_inventory):
        self.inventory = np.array(initial_inventory, dtype=float)

    def get_inventory(self):
        return self.inventory

    def check_sufficient(self, demand_vector, resource_matrix):
        """ Calculates required resources and checks if sufficient. """
        required_resources = demand_vector @ resource_matrix # Shape (M,)
        return np.all(self.inventory >= required_resources)

    def update_inventory(self, satisfied_demand_vector, resource_matrix):
        """ Updates inventory based on *satisfied* demand. """
        consumed_resources = satisfied_demand_vector @ resource_matrix
        self.inventory -= consumed_resources
        # Ensure inventory doesn't go negative due to float precision
        self.inventory = np.maximum(self.inventory, 0)
        return consumed_resources # Return what was actually consumed

    def get_satisfied_demand(self, demand_vector_D, resource_matrix_A):
         """ Determines how much demand can be satisfied given current inventory. """
         # This is a simplification. A real system might prioritize certain
         # demands or use rationing. Here, we check if *all* demand can be met.
         # If not, we assume zero sales for this simplified example.
         # A better approach involves fractional satisfaction or solving a small feasibility LP.
         required = demand_vector_D @ resource_matrix_A # Resources needed for full demand D
         if np.all(self.inventory >= required):
             return np.array(demand_vector_D) # Can satisfy all
         else:
              # Simplification: If cannot satisfy all, satisfy none.
              # TODO: Implement a more realistic partial satisfaction logic if needed.
             print(f"WARN: Insufficient inventory {self.inventory} for demand {demand_vector_D} requiring {required}. Satisfying 0.")
             return np.zeros_like(demand_vector_D)


class DemandUpdater:
    def __init__(self, products_n, price_idxs, context_list):
        self.N = products_n
        self.price_indices = price_idxs
        self.contexts = context_list
        # Posterior parameters (Beta distribution: alpha, beta)
        # Initialize with Beta(1, 1) which is Uniform(0, 1) prior
        self.posterior_params = defaultdict(lambda: {'alpha': 1.0, 'beta': 1.0}) # Key: (i, k, xi)

    def sample_demand_params(self):
        """ Samples demand probabilities from the current posterior. """
        sampled_params = {}
        for k in self.price_indices:
            for i in range(self.N):
                for xi in self.contexts:
                    params = self.posterior_params[(i, k, xi)]
                    sampled_prob = beta_dist.rvs(params['alpha'], params['beta'])
                    sampled_params[(i, k, xi)] = sampled_prob
        return sampled_params # Corresponds to theta(t) -> d(t)

    def update_posterior(self, arrived_feedback):
        """ Updates Beta posterior based on newly arrived feedback. """
        # arrived_feedback is a list of tuples:
        # (period_s, k_chosen_s, context_s, demand_vector_D_s)
        print(f"Updating posterior with {len(arrived_feedback)} arrived feedbacks.")
        for s, k, xi, D_s in arrived_feedback:
            for i in range(self.N):
                outcome = D_s[i] # 1 if purchased, 0 otherwise
                key = (i, k, xi)
                if outcome == 1:
                    self.posterior_params[key]['alpha'] += 1
                else:
                    self.posterior_params[key]['beta'] += 1

class FeedbackBuffer:
    def __init__(self):
        # Stores tuples: (arrival_time, period_s, k_chosen_s, context_s, demand_vector_D_s)
        self.buffer = []

    def add_observation(self, s, k_chosen, context, demand_vector, arrival_time):
        self.buffer.append((arrival_time, s, k_chosen, context, demand_vector))
        # Keep buffer sorted by arrival time for efficient processing
        self.buffer.sort()

    def process_arrivals(self, current_time_t):
        """ Returns feedback that arrives *before* period t starts (i.e., at t-1)"""
        arrived_feedback_details = []
        remaining_buffer = []
        for arrival_time, s, k, xi, D_s in self.buffer:
            if arrival_time <= current_time_t - 1:
                 # This feedback is now available
                arrived_feedback_details.append((s, k, xi, D_s))
            else:
                # This feedback arrives later
                remaining_buffer.append((arrival_time, s, k, xi, D_s))

        self.buffer = remaining_buffer
        return arrived_feedback_details # List of (s, k, xi, D_s) tuples


# --- Main Simulation Loop ---
def run_simulation(T, N, M, K, price_vectors, A, I_initial, contexts, context_probs, true_demand_params, use_ts_update=True):
    price_indices = list(price_vectors.keys())
    num_price_vectors = len(price_indices)

    inventory_manager = InventoryManager(I_initial)
    demand_updater = DemandUpdater(N, price_indices, contexts)
    feedback_buffer = FeedbackBuffer()

    results = {
        'revenue': np.zeros(T),
        'chosen_price_vector_idx': np.zeros(T, dtype=int),
        'actual_demand': [np.zeros(N) for _ in range(T)],
        'satisfied_demand': [np.zeros(N) for _ in range(T)],
        'inventory': [np.copy(I_initial)] * (T + 1),
        'context': [None] * T,
        'consumed_resources': [np.zeros(M) for _ in range(T)],
    }

    # --- Simulation ---
    for t in range(T):
        print(f"\n--- Period t={t} ---")
        current_inventory = inventory_manager.get_inventory()
        results['inventory'][t] = np.copy(current_inventory)
        print(f"Start Inventory: {current_inventory}")

        # 1. Process Arrived Feedback & Update Posterior
        arrived_data = feedback_buffer.process_arrivals(t)
        if arrived_data:
            demand_updater.update_posterior(arrived_data)

        # 2. Calculate Inventory Constraint c_j(t) (if using TS-Update)
        if use_ts_update:
            remaining_time = T - t
            if remaining_time > 0:
                # Ensure inventory isn't negative before division
                safe_inventory = np.maximum(current_inventory, 0)
                c_j_t = safe_inventory / remaining_time
            else:
                c_j_t = np.zeros(M) # No time left
        else:
            # TS-Fixed style constraint (using initial average)
            c_j_t = I_initial / T
        print(f"Inventory Constraint c_j(t): {c_j_t}")

        # 3. Sample Demand Parameters (from delay-updated posterior)
        sampled_demand_probs = demand_updater.sample_demand_params() # This is d(t) derived from theta(t)

        # 4. Observe Current Context (for simplified LP)
        current_context = random.choices(contexts, weights=context_probs.values(), k=1)[0]
        results['context'][t] = current_context
        print(f"Current Context: {current_context}")

        # 5. Optimize Prices (LP) - Simplified for current context
        # Objective: Maximize sum_k [ sum_i (p_ik * d_ik|xi(t)(t)) ] * x_k
        # Constraints:
        #   sum_k [ sum_i (a_ij * d_ik|xi(t)(t)) ] * x_k <= c_j(t)  (for each j in M)
        #   sum_k x_k <= 1
        #   x_k >= 0

        c = np.zeros(num_price_vectors) # Coefficients for objective function (negative for maximization)
        A_ub = np.zeros((M + 1, num_price_vectors)) # Coefficients for inequality constraints
        b_ub = np.zeros(M + 1)                # RHS for inequality constraints

        for k_idx, k in enumerate(price_indices):
            expected_revenue_k = 0
            expected_consumption_k = np.zeros(M)
            for i in range(N):
                prob_ik_xi = sampled_demand_probs.get((i, k, current_context), 0)
                expected_revenue_k += price_vectors[k][i] * prob_ik_xi
                expected_consumption_k += A[i, :] * prob_ik_xi # Resource j consumption for product i

            c[k_idx] = -expected_revenue_k # Minimize negative revenue
            A_ub[:M, k_idx] = expected_consumption_k # Resource constraints
            A_ub[M, k_idx] = 1                    # Sum of probabilities constraint

        b_ub[:M] = c_j_t
        b_ub[M] = 1

        # Bounds for x_k
        bounds = [(0, None) for _ in range(num_price_vectors)]

        # Check if inventory allows *any* action, if not, maybe choose p_infinity
        if np.all(current_inventory <= 1e-6): # Effectively zero inventory
             print("Inventory depleted. Offering p_infinity (no action).")
             x_k_solution = np.zeros(num_price_vectors)
             lp_result = None # Skip LP solve
        else:
            # Solve the LP
            lp_result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if lp_result.success:
                x_k_solution = lp_result.x
                # Normalize potential floating point errors slightly > 1
                x_k_sum = np.sum(x_k_solution)
                if x_k_sum > 1.0:
                    x_k_solution /= x_k_sum
                # Ensure non-negative
                x_k_solution = np.maximum(x_k_solution, 0)
                print(f"LP Solution (Probabilities x_k): {x_k_solution}")
            else:
                # Handle LP failure (e.g., infeasible - maybe due to tight c_j(t))
                print(f"WARN: LP failed ({lp_result.message}). Offering p_infinity.")
                x_k_solution = np.zeros(num_price_vectors) # Corresponds to p_infinity

        # 6. Select Action (Price Vector) based on LP solution
        prob_p_infinity = max(0, 1.0 - np.sum(x_k_solution))
        action_probs = list(x_k_solution) + [prob_p_infinity]

        chosen_action_idx = random.choices(price_indices + ['p_infinity'], weights=action_probs, k=1)[0]

        if chosen_action_idx == 'p_infinity':
            print("Chose p_infinity")
            k_chosen = -1 # Represent p_infinity
            price_vector_chosen = np.zeros(N)
            demand_vector_D_true = np.zeros(N, dtype=int)
            satisfied_demand_vector = np.zeros(N, dtype=int)
            revenue = 0
            consumed_resources_t = np.zeros(M)
        else:
            k_chosen = chosen_action_idx
            print(f"Chose Price Vector Index k={k_chosen}")
            results['chosen_price_vector_idx'][t] = k_chosen
            price_vector_chosen = price_vectors[k_chosen]

            # 7. Simulate True Outcome & Delay
            demand_vector_D_true = np.zeros(N, dtype=int)
            for i in range(N):
                demand_vector_D_true[i] = simulate_true_demand(i, k_chosen, current_context, true_demand_params)
            results['actual_demand'][t] = demand_vector_D_true
            print(f"True Demand D({t}): {demand_vector_D_true}")

            # 8. Determine Satisfied Demand & Update Inventory
            satisfied_demand_vector = inventory_manager.get_satisfied_demand(demand_vector_D_true, A)
            consumed_resources_t = inventory_manager.update_inventory(satisfied_demand_vector, A)

            results['satisfied_demand'][t] = satisfied_demand_vector
            results['consumed_resources'][t] = consumed_resources_t
            print(f"Satisfied Demand: {satisfied_demand_vector}")
            print(f"Consumed Resources: {consumed_resources_t}")


            revenue = np.dot(satisfied_demand_vector, price_vector_chosen)
            results['revenue'][t] = revenue
            print(f"Revenue: {revenue}")

            # 9. Simulate Delay and Add to Buffer
            delay = simulate_delay(t)
            arrival_time = t + delay
            # Store the *satisfied* demand as the observation for learning
            feedback_buffer.add_observation(t, k_chosen, current_context, satisfied_demand_vector, arrival_time)
            print(f"Observation delay: {delay}, arrival time: {arrival_time}")

    # Store final inventory
    results['inventory'][T] = inventory_manager.get_inventory()
    return results

# --- Run Simulation ---
if __name__ == "__main__":
    sim_results = run_simulation(
        T, N, M, K, price_vectors, A, I_initial, contexts, context_probs,
        true_demand_params, use_ts_update=True # Set to False for TS-Fixed style constraints
    )

    # --- Analyze Results (Example) ---
    print("\n--- Simulation Finished ---")
    total_revenue = np.sum(sim_results['revenue'])
    print(f"Total Revenue: {total_revenue:.2f}")

    # Plotting example (requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(np.cumsum(sim_results['revenue']))
        plt.title('Cumulative Revenue')
        plt.xlabel('Period t')
        plt.ylabel('Cumulative Revenue')

        plt.subplot(2, 2, 2)
        inv_array = np.array(sim_results['inventory'])
        for j in range(M):
             plt.plot(inv_array[:, j], label=f'Resource {j}')
        plt.title('Inventory Levels')
        plt.xlabel('Period t')
        plt.ylabel('Units Remaining')
        plt.legend()
        plt.ylim(bottom=-5) # Allow seeing zero clearly

        plt.subplot(2, 2, 3)
        chosen_actions = sim_results['chosen_price_vector_idx']
        # Count occurrences of each price vector (excluding p_infinity represented by -1 maybe?)
        action_counts = defaultdict(int)
        for action_idx in chosen_actions:
             action_counts[action_idx] += 1
        plt.bar(action_counts.keys(), action_counts.values())
        plt.title('Chosen Price Vector Counts')
        plt.xlabel('Price Vector Index k (-1=p_inf)')
        plt.ylabel('Frequency')


        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not found. Skipping plotting.")