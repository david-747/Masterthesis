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
                 use_real_lp: bool = True  # Add a flag to easily switch
                 ):
        print("Initializing Simulator...")
        self.total_time_periods = total_time_periods
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

        # 5. Initialize Resources for CMAB
        self.resource_consumption_matrix, self.initial_resource_inventory = self._initialize_resources()
        print(f"Resource consumption matrix shape: {self.resource_consumption_matrix.shape}")
        print(f"Initial resource inventory: {self.initial_resource_inventory}")

        # CHOOSE WHICH SOLVER TO USE
        solver_function = solve_real_lp if use_real_lp else mock_lp_solver
        solver_name = "REAL LP solver" if use_real_lp else "MOCK LP solver"

        # 6. Initialize CMAB
        self.cmab = CMAB(
            agent=self.agent,
            lp_solver_function=solver_function, # Pass the chosen solver function
            all_products=self.all_products,
            all_price_vectors_map=self.all_price_vectors_map,
            resource_consumption_matrix=self.resource_consumption_matrix,
            initial_resource_inventory=self.initial_resource_inventory,
            total_time_periods=self.total_time_periods,
            context_probabilities=None  # Not using explicit context probabilities in this simple sim
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
        # Shape: (num_products, num_resources)
        # Simple: each product consumes 1 unit of each resource
        resource_consumption_matrix = np.ones((len(self.all_products), self.num_resources))

        # Shape: (num_resources,)
        # Simple: large enough initial inventory per resource
        initial_resource_inventory = np.full((self.num_resources,),
                                             float(
                                                 self.total_time_periods * len(self.all_products) * 2))  # ensure enough
        return resource_consumption_matrix, initial_resource_inventory


    #currently simulates a very simple demand model, assumption is, that there is a base demand with an added price effect and product effect
    #higher price vector ID means higher price, so lower chance of sale
    #currently there is a product effect which simulates a certain preference for one product over another
    #this will very likely be ommited in final build
    #output is a boolean indicating if the sale was successful or not
    def _simulate_demand(self, product: Product, chosen_price_vector_id: int | None,
                         chosen_price_vector: PriceVector | None) -> bool:
        """
        Simulates if a customer purchases a product.
        This is a very basic demand model for testing.
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
        print(f"\n--- Starting Simulation for {self.total_time_periods} periods ---")
        total_sales = 0

        for t in range(self.total_time_periods):
            self.current_time_t = t
            print(f"\n--- Period t = {t} ---")

            # 1. Process Arrived Feedback
            feedback_to_process_this_period = []
            # Iterate carefully if modifying deque while iterating, or build a new one
            num_pending = len(self.pending_feedback)
            for _ in range(num_pending):
                if self.pending_feedback:
                    arrival_t, feedback_id, success = self.pending_feedback[0]
                    if arrival_t <= self.current_time_t:
                        self.pending_feedback.popleft()  # Remove from queue
                        feedback_to_process_this_period.append((feedback_id, success))
                        # print(f"  Feedback arrived for ID {feedback_id}: {'Sale' if success else 'No Sale'}")
                    else:
                        # As pending_feedback is sorted by arrival_t, we can break early
                        break

            if feedback_to_process_this_period:
                print(f"  Processing {len(feedback_to_process_this_period)} feedback items for the agent.")
                # Pass the current time to the method
                self.cmab.process_feedback_for_agent(feedback_to_process_this_period, self.current_time_t)
            else:
                print("  No feedback to process this period.")

            # 2. Determine Pricing Policy for the Period
            # print("  CMAB determining pricing policy...")
            self.cmab.determine_pricing_policy_for_period()
            # In a real scenario, you might inspect self.cmab.current_lp_solution_x_ksi_k

            # 3. Simulate Observed Context for this Period
            observed_realized_context = random.choice(self.all_contexts)
            print(f"  Observed context: {observed_realized_context}")

            # 4. Select Action (Offer Price) and Record for Feedback
            # print("  CMAB selecting action...")
            chosen_price_vector_id, product_specific_feedback_ids_map = \
                self.cmab.select_action_and_record_for_feedback(
                    observed_realized_context=observed_realized_context,
                    current_time_t=self.current_time_t
                )

            if chosen_price_vector_id is not None:
                chosen_pv_object = self.all_price_vectors_map[chosen_price_vector_id]
                print(f"  CMAB chose Price Vector ID: {chosen_price_vector_id} ({chosen_pv_object.name})")

                # 5. Simulate Demand and Queue Feedback
                if product_specific_feedback_ids_map:
                    for product_obj, feedback_id in product_specific_feedback_ids_map.items():

                        #NOTE: demand is simulated with calling below function
                        success = self._simulate_demand(product_obj, chosen_price_vector_id, chosen_pv_object)
                        if success:
                            total_sales += 1

                        #NOTE: this is simulating the feedback delay. It can be done after the simulated demand \
                        # as simplified logic is: constumer does purchase decision with delay + information gets to \
                        # system with delay
                        feedback_arrival_time = self.current_time_t + random.randint(1, self.max_feedback_delay + 1)

                        # Add to deque, maintaining sorted order by arrival_time
                        new_feedback = (feedback_arrival_time, feedback_id, success)
                        if not self.pending_feedback or feedback_arrival_time >= self.pending_feedback[-1][0]:
                            self.pending_feedback.append(new_feedback)
                        else:
                            # Insert in sorted order (simple for now, can be optimized if perf critical)
                            idx = 0
                            while idx < len(self.pending_feedback) and self.pending_feedback[idx][
                                0] < feedback_arrival_time:
                                idx += 1
                            self.pending_feedback.insert(idx, new_feedback)

                        # print(f"    Product '{product_obj.name}': Offer recorded. Demand sim: {'Sale' if success else 'No Sale'}. Feedback due at t={feedback_arrival_time}.")
            else:
                print("  CMAB chose P_Infinity (no prices offered).")

        print(f"\n--- Simulation Ended after {self.total_time_periods} periods ---")
        print(f"Total sales simulated: {total_sales}")
        print(f"Pending feedback items at end: {len(self.pending_feedback)}")

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
    simulator.run()

    # --- After the simulation, visualize the results ---
    # You can pick any context and product you want to inspect.
    # Here, we'll just pick the first ones from the lists for demonstration.

    # --- After the simulation, visualize the new summary heatmap ---
    simulator.visualize_all_learnings_heatmap()

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