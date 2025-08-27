import csv
from datetime import datetime
import os
import random
import numpy as np
from collections import deque
from scipy.stats import norm

# Assuming your other project files are in the same directory or accessible in PYTHONPATH
from CMAB import CMAB
from DelayedTSAgent import DelayedTSAgent
from MiscShipping import Product, PriceVector, Price
from Context import Context, Season, CustomerType, CommodityValue, generate_all_domain_contexts
from LPSolver import solve_real_lp


class WTPSimulator:
    """
    A simulator with a Willingness-to-Pay (WTP) model that manages its
    own feedback lifecycle to avoid modifying other classes.
    """

    def __init__(self,
                 total_time_periods: int = 100,
                 avg_arrivals_per_period: float = 2.0,
                 max_feedback_delay: int = 5,
                 num_price_options: int = 3,
                 use_ts_update: bool = True,
                 pacing_aggressiveness: float = 1.0,
                 demand_shock_config: list = None
                 ):
        print("Initializing WTP Simulator...")
        self.total_time_periods = total_time_periods
        self.avg_arrivals_per_period = avg_arrivals_per_period
        self.max_feedback_delay = max_feedback_delay
        self.num_price_options = num_price_options
        self.demand_shock_config = demand_shock_config if demand_shock_config is not None else []

        self.current_time_t = 0
        self.pending_feedback = deque()

        # --- METRICS TRACKING ---
        self.total_achieved_revenue = 0.0
        self.total_units_demanded = 0
        self.total_units_sold = 0

        # --- SIMULATOR-MANAGED FEEDBACK REGISTRY ---
        # The simulator will now manage its own registry to handle the 1-action-N-outcomes model.
        self.feedback_registry = {}
        self._feedback_counter = 0

        # 1. Create Products, Prices, Contexts
        self.all_products = self._create_products()
        self.num_products = len(self.all_products)
        self.all_price_vectors_map, self.all_price_indices = self._create_price_vectors(self.all_products)
        self.all_contexts = self._create_contexts()

        # 2. Initialize Agent and CMAB
        # The agent is instantiated here but the simulator will interact with it via the CMAB
        self.agent = DelayedTSAgent(
            all_possible_contexts=self.all_contexts,
            all_possible_products=self.all_products,
            all_possible_price_indices=self.all_price_indices
        )
        self.cmab = CMAB(
            agent=self.agent,
            lp_solver_function=solve_real_lp,
            all_products=self.all_products,
            all_price_vectors_map=self.all_price_vectors_map,
            # Resource setup will be passed to CMAB methods where needed
            resource_consumption_matrix=self._initialize_resources()[0],
            initial_resource_inventory=self._initialize_resources()[1],
            total_time_periods=total_time_periods,
            pacing_aggressiveness=pacing_aggressiveness,
            use_ts_update=use_ts_update,
            demand_scaling_factor=1.0  # Not used in WTP model, but required by CMAB
        )

        # 3. Define Ground Truth WTP & Resources
        self.true_wtp_params = self._create_ground_truth_wtp()
        self.resource_consumption_matrix, self.initial_resource_inventory = self._initialize_resources()
        self.current_inventory = np.copy(self.initial_resource_inventory)

        print(
            f"Created {self.num_products} products, {len(self.all_price_indices)} price vectors, and {len(self.all_contexts)} contexts.")
        print(f"Initial resource inventory: {self.initial_resource_inventory}")
        print("Initialization Complete.")
        print("-" * 30)

    # --- Core Setup Methods (Unchanged) ---
    def _create_products(self) -> list[Product]:
        container_types = [
            {'id': 'TEU', 'name': '20ft Standard Dry'},
            {'id': 'FEU', 'name': '40ft Standard Dry'},
            {'id': 'HC', 'name': '40ft High Cube'},
            {'id': 'REEF', 'name': '40ft Reefer (Refrigerated)'}
        ]
        return [Product(product_id=ct['id'], name=ct['name']) for ct in container_types]

    def _create_price_vectors(self, products: list[Product]) -> tuple[dict[int, PriceVector], list[int]]:
        price_vectors_map = {}
        base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
        multipliers = {
            0: {'name': 'Aggressive', 'multiplier': 0.85},
            1: {'name': 'Standard', 'multiplier': 1.0},
            2: {'name': 'Premium', 'multiplier': 1.20}
        }
        for i in range(self.num_price_options):
            info = multipliers[i]
            pv = PriceVector(vector_id=i, name=f"PV_{i}_{info['name']}")
            for prod in products:
                price = base_prices[prod.product_id] * info['multiplier']
                pv.set_price(prod, Price(amount=price, currency="USD"))
            price_vectors_map[i] = pv
        return price_vectors_map, sorted(list(price_vectors_map.keys()))

    def _create_contexts(self) -> list[Context]:
        return generate_all_domain_contexts(list(Season), list(CustomerType), list(CommodityValue))

    def _initialize_resources(self) -> tuple[np.ndarray, np.ndarray]:
        consumption = {'TEU': 1, 'FEU': 2, 'HC': 2, 'REEF': 2}
        consumption_list = [consumption[p.product_id] for p in self.all_products]
        consumption_matrix = np.array(consumption_list).reshape((self.num_products, 1))
        initial_inventory = np.array([450.0])
        return consumption_matrix, initial_inventory

    def _create_ground_truth_wtp(self) -> dict:
        wtp_params = {}
        base_wtp_means = {'TEU': 2600, 'FEU': 4600, 'HC': 4950, 'REEF': 8200}
        for context in self.all_contexts:
            context_key = context.get_key()
            wtp_params[context_key] = {}
            for product in self.all_products:
                mean = base_wtp_means[product.product_id]
                if context.season == Season.HIGH: mean *= 1.15
                if context.season == Season.LOW: mean *= 0.90
                if context.customer_type == CustomerType.RECURRING: mean *= 1.05
                if context.commodity_value == CommodityValue.HIGH: mean *= 1.10
                std_dev = mean * 0.20
                wtp_params[context_key][product.product_id] = {'mean': mean, 'std_dev': std_dev}
        return wtp_params

    # --- New & Overridden Simulation Logic ---
    def _get_current_wtp_params(self, product: Product, context: Context) -> dict:
        base_params = self.true_wtp_params[context.get_key()][product.product_id]
        current_mean = base_params['mean']
        for shock in self.demand_shock_config:
            if shock['start_time'] <= self.current_time_t < shock['end_time']:
                if product.product_id in shock.get('products', [product.product_id]):
                    current_mean *= shock['multiplier']
        return {'mean': current_mean, 'std_dev': base_params['std_dev']}

    def _simulate_customer_request(self, product: Product, context: Context, offered_price: float) -> tuple[
        bool, float]:
        wtp_params = self._get_current_wtp_params(product, context)
        customer_wtp = max(0, np.random.normal(loc=wtp_params['mean'], scale=wtp_params['std_dev']))
        return offered_price <= customer_wtp, customer_wtp

    def _process_arrived_feedback(self, feedback_items: list[tuple]):
        """
        NEW: Processes feedback using the simulator's own registry.
        """
        for feedback_id, success in feedback_items:
            action_details = self.feedback_registry.get(feedback_id)
            if action_details:
                # Directly call the agent's update method via the CMAB object
                self.cmab.agent.update_posterior(
                    context=action_details["context"],
                    product=action_details["product"],
                    price_vector_id=action_details["price_vector_id"],
                    success=success
                )
                del self.feedback_registry[feedback_id]
            # else: # Optional: Add debug print for failed lookups if needed
            #     print(f"DEBUG (Simulator): Failed to find key {feedback_id} in local registry.")

    def run(self):
        """Main simulation loop using simulator-managed feedback."""
        print(f"\n--- Starting Simulation for {self.total_time_periods} hours ---")

        for t in range(self.total_time_periods):
            self.current_time_t = t

            # 1. Process Delayed Feedback using the SIMULATOR's method
            feedback_to_process = []
            while self.pending_feedback and self.pending_feedback[0][0] <= self.current_time_t:
                _t, feedback_id, success = self.pending_feedback.popleft()
                feedback_to_process.append((feedback_id, success))
            if feedback_to_process:
                self._process_arrived_feedback(feedback_to_process)

            # 2. Determine Resource Budget for Period t
            remaining_time = self.total_time_periods - t
            if self.cmab.use_ts_update and remaining_time > 0:
                resource_constraints = np.maximum(self.current_inventory, 0) / remaining_time
            else:
                resource_constraints = self.initial_resource_inventory / self.total_time_periods

            # 3. Determine Pricing Policy via CMAB
            self.cmab.determine_pricing_policy_for_period(
                current_time_t=t,
                resource_constraints_c_j=resource_constraints
            )

            # 4. Simulate Customer Arrivals for Period t
            num_arrivals = np.random.poisson(lam=self.avg_arrivals_per_period)
            if num_arrivals == 0: continue

            observed_context = random.choice(self.all_contexts)

            # Select action via CMAB but IGNORE its feedback map.
            # The CMAB will still register its own feedback, but we won't use it.
            chosen_pv_id, _ = self.cmab.select_action_and_record_for_feedback(
                observed_realized_context=observed_context,
                current_time_t=t
            )

            if chosen_pv_id is None: continue

            chosen_pv = self.all_price_vectors_map[chosen_pv_id]

            # 5. Process each arrival, generating and recording feedback locally
            for _ in range(num_arrivals):
                interested_product = random.choice(self.all_products)
                self.total_units_demanded += 1

                # Generate a unique ID in the SIMULATOR's registry
                self._feedback_counter += 1
                feedback_id = f"sim-t{t}-customer{self._feedback_counter}"
                self.feedback_registry[feedback_id] = {
                    "context": observed_context,
                    "product": interested_product,
                    "price_vector_id": chosen_pv_id
                }

                offered_price = chosen_pv.get_price_object(interested_product).amount
                sale_successful, _ = self._simulate_customer_request(interested_product, observed_context,
                                                                     offered_price)

                # Check inventory and finalize sale
                product_idx = self.cmab.product_to_idx_map[interested_product.product_id]
                required_resources = self.resource_consumption_matrix[product_idx, :]

                final_success = False
                if sale_successful and np.all(self.current_inventory >= required_resources):
                    self.current_inventory -= required_resources
                    self.total_achieved_revenue += offered_price
                    self.total_units_sold += 1
                    final_success = True

                # Schedule feedback using the SIMULATOR's unique ID
                feedback_arrival_time = t + random.randint(1, self.max_feedback_delay + 1)
                self.pending_feedback.append((feedback_arrival_time, feedback_id, final_success))

            self.pending_feedback = deque(sorted(list(self.pending_feedback)))

        print(f"\n--- Simulation Ended ---")
        return {
            "achieved_revenue": self.total_achieved_revenue,
            "fill_rate": (self.total_units_sold / self.total_units_demanded) if self.total_units_demanded > 0 else 0
        }

    def calculate_oracle_revenue(self) -> float:
        # This method is unchanged and should work correctly
        print("--- Calculating Benchmark Optimal Revenue (WTP Oracle) ---")
        oracle_theta = {}
        for context in self.all_contexts:
            context_key = context.get_key()
            oracle_theta[context] = {}
            for product in self.all_products:
                oracle_theta[context][product] = {}
                for pv_id, pv_obj in self.all_price_vectors_map.items():
                    price = pv_obj.get_price_object(product).amount
                    wtp_params = self.true_wtp_params[context_key][product.product_id]
                    prob_sale = 1 - norm.cdf(price, loc=wtp_params['mean'], scale=wtp_params['std_dev'])
                    oracle_theta[context][product][pv_id] = prob_sale

        fixed_resource_constraints = self.initial_resource_inventory / self.total_time_periods

        benchmark_lp_solution = solve_real_lp(
            sampled_theta_t=oracle_theta,
            resource_constraints_c_j=fixed_resource_constraints,
            all_contexts=self.all_contexts,
            all_products=self.all_products,
            all_price_indices=self.all_price_indices,
            all_price_vectors_map=self.all_price_vectors_map,
            resource_consumption_matrix_A_ij=self.resource_consumption_matrix,
            context_probabilities=None,
            demand_scaling_factor=self.avg_arrivals_per_period,
            product_to_idx_map=self.cmab.product_to_idx_map
        )
        total_lp_value = 0
        for context, policy in benchmark_lp_solution.items():
            context_revenue = 0
            for pv_id, prob_x in policy.items():
                pv_expected_revenue = 0
                for product in self.all_products:
                    price = self.all_price_vectors_map[pv_id].get_price_object(product).amount
                    expected_revenue_per_arrival = price * oracle_theta[context][product][pv_id]
                    pv_expected_revenue += expected_revenue_per_arrival
                context_revenue += pv_expected_revenue * prob_x
            total_lp_value += context_revenue
        avg_optimal_revenue_per_period = total_lp_value / len(self.all_contexts)
        total_benchmark_revenue = avg_optimal_revenue_per_period * self.avg_arrivals_per_period * self.total_time_periods

        print(f"Total Benchmark Revenue (Oracle): ${total_benchmark_revenue:,.2f}")
        return total_benchmark_revenue


# --- Main Execution Block (Unchanged) ---
if __name__ == '__main__':
    output_filename = "../wtp_simulation_results.csv"
    num_simulation_runs = 10
    sim_config = {
        "total_time_periods": 504,
        "avg_arrivals_per_period": 2.5,
        "max_feedback_delay": 24,
        "num_price_options": 3,
        "use_ts_update": True,
        "pacing_aggressiveness": 0.8,
        "demand_shock_config": [
            {
                "start_time": 200,
                "end_time": 350,
                "multiplier": 1.5,
                "products": ["REEF", "HC"]
            }
        ]
    }
    batch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = [
        'timestamp', 'run_number', 'total_time_periods', 'avg_arrivals_per_period',
        'max_feedback_delay', 'use_ts_update', 'pacing_aggressiveness',
        'oracle_revenue', 'achieved_revenue', 'regret', 'fill_rate'
    ]
    file_exists = os.path.isfile(output_filename)
    with open(output_filename, 'a+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i in range(num_simulation_runs):
            print(f"\n\n--- Starting WTP Simulation Run #{i + 1}/{num_simulation_runs} ---")
            simulator = WTPSimulator(**sim_config)
            oracle_revenue = simulator.calculate_oracle_revenue()
            sim_results = simulator.run()
            achieved_revenue = sim_results['achieved_revenue']
            fill_rate = sim_results['fill_rate']
            regret = oracle_revenue - achieved_revenue

            print("\n--- Run Summary ---")
            print(f"Oracle Revenue  : ${oracle_revenue:12,.2f}")
            print(f"Achieved Revenue: ${achieved_revenue:12,.2f}")
            print(f"Regret          : ${regret:12,.2f}")
            print(f"Fill Rate       : {fill_rate:12.2%}")

            log_data = {
                'timestamp': batch_timestamp, 'run_number': i + 1,
                'total_time_periods': sim_config['total_time_periods'],
                'avg_arrivals_per_period': sim_config['avg_arrivals_per_period'],
                'max_feedback_delay': sim_config['max_feedback_delay'],
                'use_ts_update': sim_config['use_ts_update'],
                'pacing_aggressiveness': sim_config['pacing_aggressiveness'],
                'oracle_revenue': round(oracle_revenue, 2),
                'achieved_revenue': round(achieved_revenue, 2),
                'regret': round(regret, 2),
                'fill_rate': round(fill_rate, 4)
            }
            writer.writerow(log_data)
            csvfile.flush()

    print(f"\n\n--- All {num_simulation_runs} runs complete. Results saved to {output_filename} ---")