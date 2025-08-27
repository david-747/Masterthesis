import csv
from datetime import datetime
import os
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# Assuming your files are in the same directory or accessible in PYTHONPATH
from CMAB import CMAB
from DelayedTSAgent import DelayedTSAgent
from MiscShipping import Product, PriceVector, Price
from Context import Context, Season, CustomerType, CommodityValue, generate_all_domain_contexts

# Import the real LP solver from its new file
from LPSolver import solve_real_lp

# --- Mock LP Solver ---
# This is a placeholder for a real LP solver.
# It needs to return a policy: {context_obj: {price_vector_id: probability_x_ksi_k}}
#NOTE: mock solver included for testing but should be ommited in final build
def mock_lp_solver(
        sampled_theta_t,
        resource_constraints_c_j,
        all_contexts,
        all_products,
        all_price_indices,  # List of price vector IDs
        all_price_vectors_map,
        resource_consumption_matrix_A_ij,
        context_probabilities,  # Optional P(ksi) for E_ksi in LP objective
        product_to_idx_map
) -> dict:
    """
    A mock LP solver that returns a simple policy.
    For each context, it randomly picks one price vector and assigns it probability 1.0.
    Ignores sampled_theta_t, resource constraints, etc., for simplicity in this test simulator.
    """
    lp_solution = {}
    if not all_price_indices:  # No price vectors to choose from
        for context_obj in all_contexts:
            lp_solution[context_obj] = {}  # Results in offering p_infinity
        return lp_solution

    for context_obj in all_contexts:
        # Simplistic strategy: randomly choose one price vector to offer with probability 1.0
        chosen_pv_id = random.choice(all_price_indices)
        probabilities_for_context = {pv_id: 0.0 for pv_id in all_price_indices}
        probabilities_for_context[chosen_pv_id] = 1.0
        lp_solution[context_obj] = probabilities_for_context
    return lp_solution


class Simulator:
    def __init__(self,
                 total_time_periods: int = 100,
                 num_products: int = 2,
                 num_price_options_per_product: int = 2,
                 max_feedback_delay: int = 5,
                 num_resources: int = 1,
                 # --- NEW & MODIFIED ARGUMENTS ---
                 avg_arrivals_per_hour: float = 1.5,  # Lambda for Poisson distribution
                 arrival_decay_kappa: float = 1.5,
                 use_ts_update: bool = False,
                 pacing_aggressiveness: float = 1.0,
                 use_real_lp: bool = True  # Kept for consistency
                 ):
        print("Initializing Simulator...")
        self.total_time_periods = total_time_periods

        # --- NEW: Store the average arrival rate ---
        self.avg_arrivals_per_hour = avg_arrivals_per_hour
        self.arrival_decay_kappa = arrival_decay_kappa

        self.num_products = num_products
        self.num_price_options = num_price_options_per_product  # Simplified: total number of price vectors
        self.max_feedback_delay = max_feedback_delay
        self.num_resources = num_resources

        self.current_time_t = 0
        self.pending_feedback = deque()  # Stores (arrival_time_t, feedback_id, success_bool)

        # 1. Create Products
        self.all_products = self._create_products()
        print(f"Created {len(self.all_products)} products: {[p.name for p in self.all_products]}")

        # 2. Create Price Vectors
        self.all_price_vectors_map, self.all_price_indices = self._create_price_vectors(self.all_products)
        print(f"Created {len(self.all_price_indices)} price vectors with IDs: {self.all_price_indices}")
        # for pv_id, pv in self.all_price_vectors_map.items():
        #     print(f"  PV {pv_id}: {pv.name} - {pv.prices_per_product}")

        # 3. Create Contexts
        self.all_contexts = self._create_contexts()
        print(f"Created {len(self.all_contexts)} contexts.")
        # print(f"Example context: {self.all_contexts[0]}")

        # 4. Initialize DelayedTSAgent
        self.agent = DelayedTSAgent(
            all_possible_contexts=self.all_contexts,
            all_possible_products=self.all_products,  # Agent expects list of Product objects
            all_possible_price_indices=self.all_price_indices
        )
        print("Initialized DelayedTSAgent.")

        # 4.5 setting ground truth
        self.true_demand_theta = self._create_ground_truth_demand()

        # 5. Initialize Resources for CMAB
        self.resource_consumption_matrix, self.initial_resource_inventory = self._initialize_resources()
        print(f"Resource consumption matrix shape: {self.resource_consumption_matrix.shape}")
        print(f"Initial resource inventory: {self.initial_resource_inventory}")

        # --- ADD THESE LINES in __init__ ---
        self.current_day = -1  # Initialize to -1 to ensure day 0 is triggered
        self.daily_resource_bucket = np.zeros_like(self.initial_resource_inventory)
        # ---

        # --- ADD THIS LINE ---
        self.current_inventory = np.copy(self.initial_resource_inventory)
        # ---

        # CHOOSE WHICH SOLVER TO USE
        solver_function = solve_real_lp if use_real_lp else mock_lp_solver
        solver_name = "REAL LP solver" if use_real_lp else "MOCK LP solver"

        self.use_ts_update = use_ts_update
        self.demand_scaling_factor = 2.0  # Centralized scaling factor

        # 6. Initialize CMAB
        self.cmab = CMAB(
            agent=self.agent,
            lp_solver_function=solver_function, # Pass the chosen solver function
            all_products=self.all_products,
            all_price_vectors_map=self.all_price_vectors_map,
            resource_consumption_matrix=self.resource_consumption_matrix,
            initial_resource_inventory=self.initial_resource_inventory,
            total_time_periods=self.total_time_periods,
            context_probabilities=None,  # Not using explicit context probabilities in this simple sim
            demand_scaling_factor=self.demand_scaling_factor,
            pacing_aggressiveness=pacing_aggressiveness,  # <-- PASS THE ARGUMENT HERE
            use_ts_update=use_ts_update  # <-- Use the argument passed to Simulator
        )
        print(f"Initialized CMAB with {solver_name}.")
        print("-" * 30)

    #currently creates "products" from the number of products specified in the config
    def _create_products(self) -> list[Product]:

        print("Creating specific container types...")

        # Define the standard container types (our "products")
        # You can easily add or remove types from this list.
        container_types = [
            {'id': 'TEU', 'name': '20ft Standard Dry'},
            {'id': 'FEU', 'name': '40ft Standard Dry'},
            {'id': 'HC', 'name': '40ft High Cube'},
            {'id': 'REEF', 'name': '40ft Reefer (Refrigerated)'}
        ]

        products = []
        for container in container_types:
            products.append(Product(product_id=container['id'], name=container['name']))

            # The number of products is now determined by the length of the list above
        self.num_products = len(products)

        return products

    #currently creates "price_vectors" from the number of products specified in the config and the number of price options specified in the config
    def _create_price_vectors(self, products: list[Product]) -> tuple[dict[int, PriceVector], list[int]]:
        price_vectors_map = {}

        # 1. Define realistic base prices for each container type (product_id)
        # These are your internal reference or "standard" prices.
        base_container_prices = {
            'TEU': 2500.0,  # 20ft Standard
            'FEU': 4500.0,  # 40ft Standard
            'HC': 4800.0,  # 40ft High Cube
            'REEF': 8000.0  # 40ft Reefer has a premium price
        }

        # 2. Define the pricing levels/strategies you want the agent to explore.
        # These multipliers will be applied to the base prices.
        # The number of levels is determined by 'self.num_price_options' from the config.
        # Example for num_price_options = 3: [Aggressive, Standard, Premium]
        price_level_multipliers = {
            0: {'name': 'Aggressive', 'multiplier': 0.85},  # 15% discount
            1: {'name': 'Standard', 'multiplier': 1.0},  # Base price
            2: {'name': 'Premium', 'multiplier': 1.20}  # 20% premium
            # Add more levels if self.num_price_options is higher
        }

        # Ensure we have enough defined levels for the configuration
        if self.num_price_options > len(price_level_multipliers):
            raise ValueError(
                f"num_price_options is {self.num_price_options}, but only {len(price_level_multipliers)} price levels are defined.")

        # 3. Create a price vector for each pricing level
        for i in range(self.num_price_options):
            level_info = price_level_multipliers[i]
            pv_id = i
            pv_name = f"PV_{i}_{level_info['name']}"
            multiplier = level_info['multiplier']

            price_vector = PriceVector(vector_id=pv_id, name=pv_name)

            # For this price vector, calculate the price for EACH container type
            for prod in products:
                if prod.product_id not in base_container_prices:
                    print(f"Warning: Product ID '{prod.product_id}' not found in base_container_prices map. Skipping.")
                    continue

                base_price = base_container_prices[prod.product_id]
                final_price = base_price * multiplier

                price_vector.set_price(prod, Price(amount=final_price, currency="USD"))

            price_vectors_map[pv_id] = price_vector

        price_indices = sorted(list(price_vectors_map.keys()))
        if not price_indices and self.num_price_options > 0:
            raise Exception("Price indices are empty despite num_price_options > 0")

        return price_vectors_map, price_indices

    #currently creates all contexts from the context enum
    def _create_contexts(self) -> list[Context]:
        # Using all enum members to generate all combinations
        all_seasons = list(Season)
        all_customer_types = list(CustomerType)
        all_commodity_values = list(CommodityValue)
        return generate_all_domain_contexts(all_seasons, all_customer_types, all_commodity_values)

    #idea is to have an array as inventory and a matrix for resource consumption
    def _initialize_resources(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        # Shape: (num_products, num_resources)
        # Simple: each product consumes 1 unit of each resource
        resource_consumption_matrix = np.ones((len(self.all_products), self.num_resources))

        # Shape: (num_resources,)
        # Simple: large enough initial inventory per resource
        initial_resource_inventory = np.full((self.num_resources,),
                                             float(
                                                 self.total_time_periods * len(self.all_products) * 2))  # ensure enough
        return resource_consumption_matrix, initial_resource_inventory
        '''
        """
            Initializes a realistic, scarce inventory and consumption matrix.
            The single resource is "TEU slots" on a ship.
            """
        # 1. Define the consumption of "TEU slots" for each product type.
        # The order must match the order in self.all_products.
        # Our order is: 'TEU', 'FEU', 'HC', 'REEF'
        consumption_per_product = {
            'TEU': 1,  # A 20ft container takes 1 TEU slot
            'FEU': 2,  # A 40ft container takes 2 TEU slots
            'HC': 2,  # A 40ft High Cube takes 2 TEU slots
            'REEF': 2  # A 40ft Reefer takes 2 TEU slots
        }

        # Create the consumption matrix based on the product order
        resource_consumption_list = [consumption_per_product[p.product_id] for p in self.all_products]
        resource_consumption_matrix = np.array(resource_consumption_list).reshape((self.num_products, 1))

        # 2. Set the initial inventory based on your realistic constraints.
        # Total capacity for spot customers is 450 TEU slots.
        initial_resource_inventory = np.array([450.0])

        return resource_consumption_matrix, initial_resource_inventory

    def _create_ground_truth_demand(self):
        """
        Creates the true, underlying demand probabilities for the entire simulation.
        This is the "oracle" information that the agent does not know.
        """
        true_demand_theta = {}
        for context in self.all_contexts:
            true_demand_theta[context] = {}
            for product in self.all_products:
                true_demand_theta[context][product] = {}
                for pv_id in self.all_price_indices:
                    # Example: Define a realistic, but unknown-to-the-agent, demand.
                    # Here, higher price vector IDs lead to lower demand.
                    # You can make this as complex as you want.
                    base_prob = 0.7
                    price_effect = -0.2 * pv_id
                    context_effect = 0.1 if "HIGH" in str(context.season) else 0

                    true_prob = base_prob + price_effect + context_effect
                    # Clamp probability between 0.05 and 0.95
                    true_demand_theta[context][product][pv_id] = max(0.05, min(0.95, true_prob))

        return true_demand_theta

    #currently simulates a very simple demand model, assumption is, that there is a base demand with an added price effect and product effect
    #higher price vector ID means higher price, so lower chance of sale
    #currently there is a product effect which simulates a certain preference for one product over another
    #this will very likely be ommited in final build
    #output is a boolean indicating if the sale was successful or not

    def _simulate_demand(self, product: Product, chosen_price_vector_id: int, context: Context) -> bool:
        #Note: Old code
        """
        def _simulate_demand(self, product: Product, chosen_price_vector_id: int | None,
                         chosen_price_vector: PriceVector | None) -> bool:
        Simulates if a customer purchases a product.
        This is a very basic demand model for testing.
        """
        """
        if chosen_price_vector_id is None or chosen_price_vector is None:
            return False  # No offer, no sale

        # Example: Higher price vector ID means higher price, so lower chance of sale
        # Product P001 might be more popular than P002 etc.
        base_sale_prob = 0.6
        price_effect = (self.num_price_options - 1 - chosen_price_vector_id) * 0.1  # Higher ID = lower prob
        #NOTE: product effect included but may be ommited in final build
        product_effect = 0.0
        if product.product_id == "P001":
            product_effect = 0.1
        elif product.product_id == "P002":
            product_effect = -0.05

        sale_prob = base_sale_prob + price_effect + product_effect
        sale_prob = max(0.05, min(0.95, sale_prob))  # Clamp probability
        
        return random.random() < sale_prob
        """

        """
            Simulates if a customer purchases a product based on the GROUND TRUTH demand.
            """

        #Note: below is the code that simulates at most 1 demand for each product
        '''
        if chosen_price_vector_id is None:
            return False

        # Get the true probability from our ground truth model
        true_prob = self.true_demand_theta[context][product][chosen_price_vector_id]

        return random.random() < true_prob
        '''

        """
                Simulates the quantity of a product a customer demands based on the GROUND TRUTH demand,
                using a Poisson distribution.
                Returns the quantity demanded (an integer >= 0).
                """
        if chosen_price_vector_id is None:
            return 0  # No offer, no demand

        # The true_prob from the ground truth model can be used to model the
        # AVERAGE quantity demanded in a Poisson distribution.
        # Let's say a high probability (e.g., 0.9) corresponds to an average demand of 2 units.
        # We can set lambda = true_prob * scaling_factor. A scaling factor of 2.0 is a reasonable start.
        #scaling_factor = 2.0
        true_prob = self.true_demand_theta[context][product][chosen_price_vector_id]

        # The rate (lambda) for the Poisson distribution.
        #lambda_rate = true_prob * scaling_factor
        lambda_rate = true_prob * self.demand_scaling_factor

        # Sample the demand quantity from the Poisson distribution.
        demanded_quantity = np.random.poisson(lam=lambda_rate)

        return int(demanded_quantity)



    # In Simulator.py, maybe in a new "run_and_evaluate" method

    def run_and_evaluate(self):
        print("--- Calculating Benchmark Optimal Revenue (Oracle) ---")
        fixed_resource_constraints = self.initial_resource_inventory / self.total_time_periods

        benchmark_lp_solution = solve_real_lp(
            sampled_theta_t=self.true_demand_theta,
            resource_constraints_c_j=fixed_resource_constraints,
            all_contexts=self.all_contexts,
            all_products=self.all_products,
            all_price_indices=self.all_price_indices,
            all_price_vectors_map=self.all_price_vectors_map,
            resource_consumption_matrix_A_ij=self.resource_consumption_matrix,
            context_probabilities=None,
            demand_scaling_factor=self.demand_scaling_factor,  # Pass the factor
            product_to_idx_map=self.cmab.product_to_idx_map
        )

        total_lp_value = 0
        for context, policy in benchmark_lp_solution.items():
            context_revenue = 0
            for pv_id, prob in policy.items():
                revenue_per_pv = 0
                for product in self.all_products:
                    price = self.all_price_vectors_map[pv_id].get_price_object(product).amount
                    #true_demand = self.true_demand_theta[context][product][pv_id]
                    #revenue_per_pv += price * true_demand

                    # --- CORRECTED BENCHMARK LOGIC ---
                    true_prob = self.true_demand_theta[context][product][pv_id]
                    expected_quantity = true_prob * self.demand_scaling_factor
                    revenue_per_pv += price * expected_quantity
                    # --- END CORRECTION ---

                context_revenue += revenue_per_pv * prob
            total_lp_value += context_revenue

        avg_optimal_revenue_per_period = total_lp_value / len(self.all_contexts)
        total_benchmark_revenue = avg_optimal_revenue_per_period * self.total_time_periods
        print(f"Total Benchmark Revenue (Oracle): ${total_benchmark_revenue:,.2f}")

        # --- THIS IS THE CORRECTED LOGIC ---
        # 1. Call run() ONCE and store the result.
        total_achieved_revenue = self.run()

        # 2. Calculate regret and performance using the stored result.
        regret = total_benchmark_revenue - total_achieved_revenue
        if total_benchmark_revenue > 0:
            performance_percentage = (total_achieved_revenue / total_benchmark_revenue) * 100
        else:
            performance_percentage = 0.0

        print(f"\n--- Performance Summary ---")
        print(f"Total Achieved Revenue: ${total_achieved_revenue:,.2f}")
        print(f"Total Regret: ${regret:,.2f}")
        print(f"Percentage of Optimal Revenue Achieved: {performance_percentage:.2f}%")

        # 3. Return the dictionary of results.
        return {
            "benchmark_revenue": total_benchmark_revenue,
            "achieved_revenue": total_achieved_revenue,
            "regret": regret,
            "performance_percentage": performance_percentage
        }


    #simulator that runs for total_time_periods
    #Step 1: first, pending feedback is checked, where the queue is iterated from the front (i.e. oldest first)
    #as feedback list ist ordered, the loop terminates the moment current_time_t is greater than arrival_t
    #all relevant feedback to process is passed to the CMAB which then passes it to the agent to get a better prior
    # for the beta distribution
    #Step 2: cmab is called to determine the pricing policy for the current period
    #Step 3: observed context is "observed", for now just randomly chosen from all contexts
    #Step 4: cmab selects and action (i.e. price vector) and passes vector id + prepared product_feedback map
    #Step 5: consequences of chosen action are simulated
    def run(self):
        print(f"\n--- Starting Simulation for {self.total_time_periods} hours ---")
        total_achieved_revenue = 0

        # This is the main loop over each hour (period t) in the simulation.
        for t in range(self.total_time_periods):
            self.current_time_t = t
            print(f"\n--- Hour t = {t}, Inventory: {self.current_inventory} ---")

            # 1. PROCESS FEEDBACK (Your existing logic for this is correct)
            feedback_to_process_this_period = []
            while self.pending_feedback and self.pending_feedback[0][0] <= self.current_time_t:
                _arrival_t, feedback_id, success = self.pending_feedback.popleft()
                feedback_to_process_this_period.append((feedback_id, success))

            if feedback_to_process_this_period:
                print(f"  Processing {len(feedback_to_process_this_period)} feedback items for the agent.")
                self.cmab.process_feedback_for_agent(feedback_to_process_this_period, self.current_time_t)
            else:
                print("  No feedback to process this period.")

            # --- ALIGNMENT CHANGE 1: Implement TS-Fixed/TS-Update Budgeting ---
            # This block replaces the entire "Daily Bucket Management" logic.
            # We calculate the resource constraint c_j for this specific period t.
            if self.use_ts_update:
                # TS-Update Logic from Algorithm 2 [cite: 297, 304]
                remaining_time = self.total_time_periods - self.current_time_t
                if remaining_time > 0:
                    safe_inventory = np.maximum(self.current_inventory, 0)
                    resource_constraints_c_j = safe_inventory / remaining_time
                else:
                    resource_constraints_c_j = np.zeros_like(self.current_inventory)
            else:
                # TS-Fixed Logic from Algorithm 1 [cite: 220, 234]
                resource_constraints_c_j = self.initial_resource_inventory / self.total_time_periods

            # --- ALIGNMENT CHANGE 2: Make ONE Decision Per Period ---
            # The following steps are now performed only ONCE per hour t, not in a loop over customers.

            # 2a. Determine Pricing Policy for the entire period t
            self.cmab.determine_pricing_policy_for_period(
                current_time_t=self.current_time_t,
                resource_constraints_c_j=resource_constraints_c_j
            )

            # 2b. Observe a context and select ONE price vector for the period
            observed_realized_context = random.choice(self.all_contexts)
            print(f"    Observed context: {observed_realized_context}")

            chosen_price_vector_id, product_specific_feedback_ids_map = \
                self.cmab.select_action_and_record_for_feedback(
                    observed_realized_context=observed_realized_context,
                    current_time_t=self.current_time_t
                )

            # 2c. Simulate the AGGREGATE outcome for the period
            if chosen_price_vector_id is not None:
                chosen_pv_object = self.all_price_vectors_map[chosen_price_vector_id]
                print(
                    f"    CMAB chose Price Vector ID: {chosen_price_vector_id} ({chosen_pv_object.name}) for this period.")

                # Your _simulate_demand function already returns a quantity, which is perfect for this.
                demands_per_product = {
                    prod: self._simulate_demand(prod, chosen_price_vector_id, observed_realized_context)
                    for prod in self.all_products
                }

                total_required_resources = np.zeros_like(self.current_inventory)
                for product_obj, quantity in demands_per_product.items():
                    if quantity > 0:
                        product_idx = self.cmab.product_to_idx_map[product_obj.product_id]
                        total_required_resources += self.resource_consumption_matrix[product_idx, :] * quantity

                # Check if aggregate demand can be met by remaining inventory
                if np.all(self.current_inventory >= total_required_resources):
                    if np.any(total_required_resources > 0):
                        demand_str = ", ".join(
                            [f'{p.product_id}: {q}' for p, q in demands_per_product.items() if q > 0])
                        print(f"    Demand vector: [{demand_str}]. Inventory sufficient, sale proceeds.")

                    self.current_inventory -= total_required_resources

                    # For each product, schedule feedback based on whether its demand was > 0
                    for product_obj, feedback_id in product_specific_feedback_ids_map.items():
                        demanded_quantity = demands_per_product.get(product_obj, 0)
                        success_for_agent = demanded_quantity > 0

                        if success_for_agent:
                            price_of_sale = chosen_pv_object.get_price_object(product_obj).amount
                            total_achieved_revenue += price_of_sale * demanded_quantity

                        feedback_arrival_time = self.current_time_t + random.randint(1, self.max_feedback_delay + 1)
                        new_feedback = (feedback_arrival_time, feedback_id, success_for_agent)
                        self.pending_feedback.append(new_feedback)

                else:  # Inventory was insufficient for the aggregate demand
                    if np.any(total_required_resources > 0):
                        print(f"    Demand existed but inventory was insufficient. No sale.")
                    # If inventory is insufficient, it's a "failure" for all products from this action
                    for product_obj, feedback_id in product_specific_feedback_ids_map.items():
                        feedback_arrival_time = self.current_time_t + random.randint(1, self.max_feedback_delay + 1)
                        new_feedback = (feedback_arrival_time, feedback_id, False)
                        self.pending_feedback.append(new_feedback)

                # Sort the pending feedback queue after adding new items to keep it ordered
                if feedback_to_process_this_period:  # Only sort if we added something
                    self.pending_feedback = deque(sorted(list(self.pending_feedback)))

            else:  # This handles the case where p_infinity was chosen
                print("    CMAB chose P_Infinity (no prices offered) for this period.")

        print(f"\n--- Simulation Ended after {self.total_time_periods} hours ---")
        return total_achieved_revenue

    # Place this method inside the Simulator class, for example, after the run() method.
    def visualize_agent_beliefs(self, context_to_show, product_to_show):
        """
        Visualizes the agent's learned Beta distributions for a specific
        product and context, comparing all available price vectors.
        """
        print("\n--- Visualizing Agent Beliefs ---")
        print(f"Product: {product_to_show.name} ({product_to_show.product_id})")
        print(f"Context: {context_to_show}")
        print("------------------------------------")

        plt.figure(figsize=(12, 7))

        x = np.linspace(0, 1, 500)

        for pv_id, pv_object in self.all_price_vectors_map.items():
            try:
                # --- THIS IS THE FIX ---
                # We must use the same stable key here that the agent uses internally.
                context_key = context_to_show.get_key()
                product_id = product_to_show.product_id

                # The key we look up must match the key used in agent.update_posterior()
                lookup_key = (context_key, product_id, pv_id)

                alpha, beta = self.agent.posterior_params.get(lookup_key, (1, 1))

                # This is the diagnostic print you can remove after it works.
                print(f"  - Plotting for {pv_object.name}: Found alpha={alpha}, beta={beta}")

            except AttributeError:
                print("Error: Could not find 'posterior_params' on the agent.")
                return

            y = beta_dist.pdf(x, alpha, beta)
            mean_prob = alpha / (alpha + beta)

            plt.plot(x, y, label=f'{pv_object.name} | α={alpha:.2f}, β={beta:.2f} | Mean={mean_prob:.2f}')

        plt.title(f'Agent Beliefs for Product "{product_to_show.name}"\nin Context "{context_to_show}"')
        plt.xlabel("Purchase Probability (θ)")
        plt.ylabel("Probability Density")
        plt.legend(title="Price Vector | Parameters | Expected Value")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(0, 1)
        plt.ylim(bottom=0)
        plt.show()

    def visualize_all_learnings_heatmap(self):
        """
        Generates a set of heatmaps to visualize the agent's learned mean
        purchase probability for every product-context-price combination.
        """
        print("\n--- Visualizing All Learned Beliefs (Heatmap Summary) ---")

        # Get the labels for the axes from our simulation setup
        # Ensure consistent ordering
        contexts = sorted(self.all_contexts, key=lambda c: c.get_key())
        price_vectors = sorted(self.all_price_vectors_map.values(), key=lambda pv: pv.vector_id)

        context_labels = [c.get_key() for c in contexts]
        price_labels = [pv.name for pv in price_vectors]

        # Generate one heatmap for each product
        for product_obj in self.all_products:
            # Create a data matrix to hold the mean probabilities for this product
            # Shape: (num_contexts, num_price_vectors)
            data_matrix = np.zeros((len(contexts), len(price_vectors)))

            # Populate the matrix with the agent's learned means
            for i, context in enumerate(contexts):
                for j, pv in enumerate(price_vectors):
                    context_key = context.get_key()
                    product_id = product_obj.product_id
                    pv_id = pv.vector_id

                    # The key to look up beliefs in the agent's dictionary
                    lookup_key = (context_key, product_id, pv_id)

                    alpha, beta = self.agent.posterior_params.get(lookup_key, (1, 1))

                    # Calculate the mean of the Beta(alpha, beta) distribution
                    mean_prob = alpha / (alpha + beta)
                    data_matrix[i, j] = mean_prob

            # Now, create the plot for the current product
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(data_matrix, cmap="viridis", vmin=0, vmax=1)

            # Set up the ticks and labels for the axes
            ax.set_xticks(np.arange(len(price_labels)))
            ax.set_yticks(np.arange(len(context_labels)))
            ax.set_xticklabels(price_labels)
            ax.set_yticklabels(context_labels)

            # Rotate the x-axis labels to prevent them from overlapping
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(len(context_labels)):
                for j in range(len(price_labels)):
                    mean_val = data_matrix[i, j]
                    # Change text color to white for dark backgrounds
                    text_color = "w" if mean_val < 0.5 else "k"
                    ax.text(j, i, f"{mean_val:.2f}", ha="center", va="center", color=text_color)

            # Add a colorbar and title
            fig.colorbar(im, ax=ax, label="Mean Purchase Probability")
            ax.set_title(f"Agent's Learned Mean Purchase Probability for: {product_obj.name}")

            fig.tight_layout()
            plt.show()


# --- Main execution ---
if __name__ == '__main__':

    '''
    # Configuration for the simulation
    sim_config = {
        "total_time_periods": 50,  # Number of periods to run the simulation
        "num_products": 2,  # Number of distinct products
        "num_price_options_per_product": 3,  # Number of price vectors (e.g. low, medium, high global price levels)
        "max_feedback_delay": 3,  # Maximum delay (in periods) for feedback to arrive
        "num_resources": 1  # Number of resource types
    }

    simulator = Simulator(
        total_time_periods=sim_config["total_time_periods"],
        num_products=sim_config["num_products"],
        num_price_options_per_product=sim_config["num_price_options_per_product"],
        max_feedback_delay=sim_config["max_feedback_delay"],
        num_resources=sim_config["num_resources"],
        use_real_lp=True  # Set to True to use your new solver
    )

    #simulator.run()
    simulator.run_and_evaluate()  # <--- CORRECTED LINE

    # --- After the simulation, visualize the results ---
    # You can pick any context and product you want to inspect.
    # Here, we'll just pick the first ones from the lists for demonstration.

    # --- After the simulation, visualize the new summary heatmap ---
    simulator.visualize_all_learnings_heatmap()
    '''

    # --- Configuration for this Batch of Runs ---
    output_filename = "../simulation_results.csv"
    num_simulation_runs = 10
    sim_config = {
        "total_time_periods": 504, #currently assuming a booking window of 3 weeks: t = 3*7*24 = 504
        "num_products": 4,
        "num_price_options_per_product": 3,
        "max_feedback_delay": 3,
        "num_resources": 1,
        "pacing_aggressiveness": 0.75  # <-- Tune this value. > 1.0 is more aggressive.
    }

    # Generate a single timestamp for this entire batch of runs
    batch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Define the headers for your CSV file
    fieldnames = [
        'timestamp', 'run_number', 'total_time_periods', 'num_products',
        'num_price_options_per_product', 'max_feedback_delay', 'num_resources',
        'pacing_aggressiveness',  # <-- ADD THIS FIELD
        'benchmark_revenue', 'achieved_revenue', 'regret', 'performance_percentage'
    ]

    # Check if the file exists to decide whether to write the header
    file_exists = os.path.isfile(output_filename)

    # Use 'a+' to append to the file. 'newline=""' is important for csv writer.
    with open(output_filename, 'a+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is new, write the header row
        if not file_exists:
            writer.writeheader()

        # --- Loop Through Simulation Runs ---
        for i in range(num_simulation_runs):
            print(f"\n\n--- Starting Simulation Run #{i + 1}/{num_simulation_runs} ---")

            # Create a new simulator instance for each run to ensure it's fresh
            #simulator = Simulator(**sim_config, use_real_lp=True, use_ts_update=True)

            # --- THIS IS THE KEY FIX ---
            # Create a new simulator instance for each run to ensure it's fresh
            simulator = Simulator(**sim_config, use_real_lp=True, use_ts_update=True)
            # ---

            # Get the dictionary of results
            results = simulator.run_and_evaluate()

            if results is not None:
                # Prepare the data row for the CSV
                log_data = {
                    'timestamp': batch_timestamp,
                    'run_number': i + 1,
                    **sim_config,  # Unpacks the sim_config dict into the log_data
                    **results  # Unpacks the results dict into the log_data
                }
                writer.writerow(log_data)
                csvfile.flush()  # Ensure data is written to disk immediately

    print(f"\n\n--- All {num_simulation_runs} runs complete. Results saved to {output_filename} ---")

    '''
    num_simulation_runs = 100  # Run the experiment 100 times
    all_performance_percentages = []

    for i in range(num_simulation_runs):
        print(f"\n\n--- Starting Simulation Run #{i + 1}/{num_simulation_runs} ---")
        # Create a new simulator instance for each run to reset the state
        sim_config = {
            "total_time_periods": 50,
            "num_products": 4,  # Make sure this matches your setup
            "num_price_options_per_product": 3,
            "max_feedback_delay": 3,
            "num_resources": 1
        }
        simulator = Simulator(**sim_config, use_real_lp=True)

        # This will run the full evaluation and return the performance
        performance = simulator.run_and_evaluate()
        if performance is not None:
            all_performance_percentages.append(performance)

    if all_performance_percentages:
        average_performance = sum(all_performance_percentages) / len(all_performance_percentages)
        print(f"\n\n--- FINAL RESULTS (after {num_simulation_runs} runs) ---")
        print(f"Average Percentage of Optimal Revenue Achieved: {average_performance:.2f}%")
    '''
    '''
    if simulator.all_contexts and simulator.all_products:
        context_to_inspect = simulator.all_contexts[0]
        product_to_inspect = simulator.all_products[0]  # e.g., '20ft Standard Dry' (TEU)

        simulator.visualize_agent_beliefs(
            context_to_show=context_to_inspect,
            product_to_show=product_to_inspect
        )

        # You could even visualize another product for comparison
        product_to_inspect_2 = simulator.all_products[1]  # e.g., '40ft Standard Dry' (FEU)
        simulator.visualize_agent_beliefs(
            context_to_show=context_to_inspect,
            product_to_show=product_to_inspect_2
        )
        '''