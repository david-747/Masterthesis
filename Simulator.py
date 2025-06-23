import random
import numpy as np
from collections import deque

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
        products = []
        for i in range(self.num_products):
            products.append(Product(product_id=f"P{i + 1:03}", name=f"Product {i + 1}"))
        return products

    #currently creates "price_vectors" from the number of products specified in the config and the number of price options specified in the config
    def _create_price_vectors(self, products: list[Product]) -> tuple[dict[int, PriceVector], list[int]]:
        price_vectors_map = {}
        # Create self.num_price_options distinct price vectors
        # Each vector will assign a price to all products
        for i in range(self.num_price_options):
            pv_id = i  # Price vector ID
            pv_name = f"PV_{i}"
            # Vary price based on pv_id (e.g., lower id = lower price)
            base_price = 100.0 + i * 50.0
            price_vector = PriceVector(vector_id=pv_id, name=pv_name)
            for p_idx, prod in enumerate(products):
                # Make product prices slightly different within the same vector
                product_price_amount = base_price + p_idx * 10.0
                price_vector.set_price(prod, Price(amount=product_price_amount, currency="USD"))
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
    #Step 4:
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
                self.cmab.process_feedback_for_agent(feedback_to_process_this_period)
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
                        success = self._simulate_demand(product_obj, chosen_price_vector_id, chosen_pv_object)
                        if success:
                            total_sales += 1

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