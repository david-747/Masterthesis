import csv
from datetime import datetime
import os
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Assuming your files are in the same directory or accessible in PYTHONPATH
from CMAB import CMAB
from DelayedTSAgent import DelayedTSAgent
from MiscShipping import Product, PriceVector, Price
from Context import Context, Season, CustomerType, CommodityValue, generate_all_domain_contexts
from LPSolver import solve_real_lp


class WtpScenarioSimulator:
    """
    A simulator designed to work with a fixed customer arrival scenario where each
    customer has a maximum Willingness-To-Pay (WTP).

    The Oracle in this simulator has perfect foresight: it knows all customer
    arrivals and their WTP in advance and calculates the maximum possible revenue
    by cherry-picking the most profitable customers subject to capacity constraints.

    The Agent operates without knowing the WTP and must learn pricing strategies
    to maximize revenue. A customer purchase occurs if the agent's offered price
    is less than or equal to the customer's maximum WTP.
    """

    def __init__(self,
                 customer_scenario_path: str,
                 num_products: int,
                 num_price_options_per_product: int,
                 max_feedback_delay: int,
                 num_resources: int,
                 use_ts_update: bool = False,
                 pacing_aggressiveness: float = 0.5,
                 use_real_lp: bool = True,
                 verbose: bool = False,
                 metrics_csv_path: str | None = "metrics_log_wtp.csv"
                 ):
        print("Initializing WtpScenarioSimulator...")

        # --- Store configurations ---
        self.customer_scenario_path = customer_scenario_path
        self.pacing_aggressiveness = pacing_aggressiveness
        self.use_ts_update = use_ts_update
        self.verbose = verbose
        self.metrics_csv_path = metrics_csv_path
        self.metrics_records: list[dict] = []

        self.num_products = num_products
        self.num_price_options = num_price_options_per_product
        self.max_feedback_delay = max_feedback_delay
        self.num_resources = num_resources

        # --- Common components ---
        self.all_products = self._create_products()
        self.all_price_vectors_map, self.all_price_indices = self._create_price_vectors(self.all_products)
        self.all_contexts = self._create_contexts()
        self.resource_consumption_matrix, self.initial_resource_inventory = self._initialize_resources()
        self.product_to_idx_map = {product.product_id: i for i, product in enumerate(self.all_products)}
        self.solver_function = solve_real_lp if use_real_lp else None
        self.demand_scaling_factor = 1.0

        # --- Load the complete customer arrival schedule with WTP ---
        self.arrival_schedule = self._load_customer_arrivals(self.customer_scenario_path)
        self.total_time_periods = 0
        if self.arrival_schedule:
            self.total_time_periods = max(d['t'] for d in self.arrival_schedule) + 1
        else:
            raise ValueError("Customer scenario could not be loaded or is empty.")

        self.arrivals_by_time = {t: [] for t in range(self.total_time_periods)}
        for arrival in self.arrival_schedule:
            self.arrivals_by_time[arrival['t']].append(arrival)

        print(
            f"Loaded scenario '{os.path.basename(customer_scenario_path)}' with {self.total_time_periods} periods and {len(self.arrival_schedule)} total arrivals.")

        # --- State variables (reset for each run) ---
        self.agent = None
        self.cmab = None
        self.current_inventory = None
        self.pending_feedback = None
        self.current_time_t = 0

    def _reset_for_new_run(self):
        """Resets the state of the simulator for a fresh run."""
        self.current_inventory = np.copy(self.initial_resource_inventory)
        self.pending_feedback = deque()
        self.current_time_t = 0
        # Do not clear metrics here so we can log both oracle and agent runs into one file.
        self.agent = DelayedTSAgent(
            all_possible_contexts=self.all_contexts,
            all_possible_products=self.all_products,
            all_possible_price_indices=self.all_price_indices
        )
        self.cmab = CMAB(
            agent=self.agent,
            lp_solver_function=self.solver_function,
            all_products=self.all_products,
            all_price_vectors_map=self.all_price_vectors_map,
            resource_consumption_matrix=self.resource_consumption_matrix,
            initial_resource_inventory=self.initial_resource_inventory,
            total_time_periods=self.total_time_periods,
            demand_scaling_factor=self.demand_scaling_factor,
            pacing_aggressiveness=self.pacing_aggressiveness,
            use_ts_update=self.use_ts_update
        )

    def _load_customer_arrivals(self, filepath: str) -> list[dict]:
        """Loads arrivals from a CSV with t, arrival_idx, max_wtp, and product quantities."""
        arrivals = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                product_ids = [p.product_id for p in self.all_products]
                for row in reader:
                    bundle = {pid: int(row[pid]) for pid in product_ids}
                    arrivals.append({
                        't': int(row['t']),
                        'arrival_idx': int(row['arrival_idx']),
                        'max_wtp': float(row['max_wtp']),
                        'bundle': bundle
                    })
            return arrivals
        except FileNotFoundError:
            print(f"Error: Customer scenario file not found at {filepath}")
            return []
        except Exception as e:
            print(f"An error occurred while loading the scenario file: {e}")
            return []

    def run(self):
        """Runs the simulation with the learning agent, returns total revenue."""
        if self.verbose: print(f"\n--- Running Learning Agent Simulation ---")
        total_achieved_revenue = 0
        cumulative_revenue_over_time = []

        total_arrivals_in_sim = sum(len(v) for v in self.arrivals_by_time.values())
        remaining_arrivals_count = total_arrivals_in_sim

        for t in range(self.total_time_periods):
            self.current_time_t = t
            arrivals_this_period = self.arrivals_by_time[t]
            if self.verbose: print(
                f"--- Agent Hour t = {t}, Inventory: {self.current_inventory}, Arrivals: {len(arrivals_this_period)} ---")

            # --- Process delayed feedback from previous periods ---
            feedback_to_process = []
            while self.pending_feedback and self.pending_feedback[0][0] <= t:
                feedback_to_process.append(self.pending_feedback.popleft())
            if feedback_to_process:
                self.cmab.process_feedback_for_agent([(fid, s) for _, fid, s in feedback_to_process], t)

            # --- Determine pricing policy for the current period ---
            if self.cmab.use_ts_update and remaining_arrivals_count > 0:
                base_budget = np.maximum(self.current_inventory, 0) / remaining_arrivals_count
                resource_constraints = base_budget * self.pacing_aggressiveness
            else:
                # Fallback pacing
                resource_constraints = self.initial_resource_inventory / self.total_time_periods

            self.cmab.determine_pricing_policy_for_period(t, resource_constraints)

            # --- Process each arrival for the current period ---
            for arrival_data in arrivals_this_period:
                observed_context = random.choice(self.all_contexts)
                bundle = arrival_data['bundle']
                max_wtp = arrival_data['max_wtp']

                # Agent selects a price vector (action)
                chosen_pv_id, feedback_map = self.cmab.select_action_and_record_for_feedback(observed_context, t)

                # Calculate the offered price for the bundle
                offered_price = 0.0
                if chosen_pv_id is not None:
                    price_vector = self.all_price_vectors_map[chosen_pv_id]
                    for pid, qty in bundle.items():
                        if qty > 0:
                            product = next(p for p in self.all_products if p.product_id == pid)
                            offered_price += price_vector.get_price_object(product).amount * qty

                # Customer decides to buy if offered price is within their WTP
                buy = (chosen_pv_id is not None) and (offered_price <= max_wtp)

                revenue_inc = 0.0
                success = False
                if buy:
                    # Check inventory
                    total_required = np.zeros_like(self.current_inventory)
                    for product_id, qty in bundle.items():
                        if qty <= 0: continue
                        idx = self.product_to_idx_map[product_id]
                        total_required += self.resource_consumption_matrix[idx, :] * qty

                    if np.all(self.current_inventory >= total_required):
                        self.current_inventory -= total_required
                        revenue_inc = offered_price  # Revenue is the price they paid
                        total_achieved_revenue += revenue_inc
                        success = True

                # Record feedback for the agent to learn from
                # --- FIX IS HERE ---
                # Only try to record feedback if a feedback map was generated
                if feedback_map is not None:
                    delay = t + random.randint(1, self.max_feedback_delay + 1)
                    for _, feedback_id in feedback_map.items():
                        self.pending_feedback.append((delay, feedback_id, success))

            remaining_arrivals_count -= len(arrivals_this_period)
            self.pending_feedback = deque(sorted(list(self.pending_feedback)))
            cumulative_revenue_over_time.append(total_achieved_revenue)

        return total_achieved_revenue, cumulative_revenue_over_time

    def _run_oracle_simulation(self):
        """
        Runs the perfect foresight oracle.
        The oracle knows all arrivals and their WTP in advance. It sorts all
        potential customers by their WTP and accepts the highest-paying ones
        until capacity is exhausted.
        """
        if self.verbose: print("--- Running Oracle Simulation for Benchmark ---")

        # 1. Calculate resource consumption for each potential arrival
        for arrival in self.arrival_schedule:
            required_resources = np.zeros(self.num_resources)
            for pid, qty in arrival['bundle'].items():
                if qty > 0:
                    idx = self.product_to_idx_map[pid]
                    required_resources += self.resource_consumption_matrix[idx, :] * qty
            arrival['resources_needed'] = required_resources

        # 2. Sort all arrivals by their maximum willingness-to-pay in descending order
        sorted_arrivals = sorted(self.arrival_schedule, key=lambda x: x['max_wtp'], reverse=True)

        # 3. "Cherry-pick" the best customers until capacity runs out
        oracle_inventory = np.copy(self.initial_resource_inventory)
        oracle_revenue = 0.0

        for arrival in sorted_arrivals:
            if np.all(oracle_inventory >= arrival['resources_needed']):
                # Accept this customer
                oracle_inventory -= arrival['resources_needed']
                oracle_revenue += arrival['max_wtp']

        if self.verbose:
            print(f"--- Oracle Simulation Finished. Benchmark Revenue: ${oracle_revenue:,.2f} ---")

        # The oracle calculation is static, so we return a constant cumulative revenue for plotting
        return oracle_revenue, [oracle_revenue] * self.total_time_periods

    def run_and_evaluate(self):
        """Calculates benchmark, runs learning agent, and returns performance metrics."""
        # Note: Oracle is run first as it doesn't have a time-based cumulative plot anymore
        total_benchmark_revenue, cumulative_benchmark_revenue = self._run_oracle_simulation()

        self._reset_for_new_run()
        total_achieved_revenue, cumulative_achieved_revenue = self.run()

        regret = total_benchmark_revenue - total_achieved_revenue
        performance_percentage = (
                                             total_achieved_revenue / total_benchmark_revenue) * 100 if total_benchmark_revenue > 0 else 0.0

        print(f"\n--- Performance Summary for Scenario: {os.path.basename(self.customer_scenario_path)} ---")
        print(f"Achieved Revenue: ${total_achieved_revenue:,.2f}")
        print(f"Benchmark (Oracle) Revenue: ${total_benchmark_revenue:,.2f}")
        print(f"Regret (Cost of Learning): ${regret:,.2f}")
        print(f"Percentage of Optimal Revenue Achieved: {performance_percentage:.2f}%")

        return {
            "benchmark_revenue": total_benchmark_revenue, "achieved_revenue": total_achieved_revenue,
            "regret": regret, "performance_percentage": performance_percentage,
            "cumulative_benchmark": cumulative_benchmark_revenue, "cumulative_achieved": cumulative_achieved_revenue
        }

    def plot_cumulative_revenue(self, results):
        """Plots the cumulative revenue of the agent vs the oracle's final revenue."""
        plt.figure(figsize=(12, 7))

        # Agent's cumulative revenue
        plt.plot(results['cumulative_achieved'], label='Agent Cumulative Revenue', color='blue')

        # Oracle's final revenue as a horizontal line
        plt.axhline(y=results['benchmark_revenue'], color='r', linestyle='--',
                    label=f"Oracle Max Revenue (${results['benchmark_revenue']:,.0f})")

        plt.title(
            f'Agent Performance vs. Perfect Foresight Oracle\nScenario: {os.path.basename(self.customer_scenario_path)}')
        plt.xlabel('Time Period (t)')
        plt.ylabel('Cumulative Revenue ($)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plot_filename = f"revenue_plot_{os.path.basename(self.customer_scenario_path).replace('.csv', '')}.png"
        plt.savefig(plot_filename)
        print(f"\nRevenue plot saved to '{plot_filename}'")
        plt.close()

    # --- HELPER METHODS (mostly unchanged) ---
    def _create_products(self) -> list[Product]:
        ct = [{'id': 'TEU', 'name': '20ft Standard Dry'}, {'id': 'FEU', 'name': '40ft Standard Dry'},
              {'id': 'HC', 'name': '40ft High Cube'}, {'id': 'REEF', 'name': '40ft Reefer (Refrigerated)'}]
        return [Product(product_id=c['id'], name=c['name']) for c in ct]

    def _create_price_vectors(self, products: list[Product]) -> tuple[dict[int, PriceVector], list[int]]:
        p_map, base = {}, {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
        mult = {0: {'n': 'Aggressive', 'm': 0.85}, 1: {'n': 'Standard', 'm': 1.0}, 2: {'n': 'Premium', 'm': 1.20}}
        for i in range(self.num_price_options):
            l = mult[i]
            pv = PriceVector(vector_id=i, name=f"PV_{i}_{l['n']}")
            for p in products:
                if p.product_id in base:
                    pv.set_price(p, Price(base[p.product_id] * l['m'], "USD"))
            p_map[i] = pv
        return p_map, sorted(list(p_map.keys()))

    def _create_contexts(self) -> list[Context]:
        return generate_all_domain_contexts(list(Season), list(CustomerType), list(CommodityValue))

    def _initialize_resources(self) -> tuple[np.ndarray, np.ndarray]:
        cons = {'TEU': 1, 'FEU': 2, 'HC': 2, 'REEF': 2}
        mat = np.array([cons[p.product_id] for p in self.all_products]).reshape((len(self.all_products), 1))
        return mat, np.array([450.0])


def generate_wtp_scenario_file(filepath="customer_scenario_wtp.csv", total_hours=504, avg_arrivals_per_hour=3.0):
    """Generates a CSV file with customer arrivals and their max WTP."""
    print(f"Generating new customer scenario file at '{filepath}'...")

    # Base prices for calculating WTP (from "Standard" price vector)
    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    product_ids = list(base_prices.keys())

    # Poisson lambdas for generating product quantities in a bundle
    poisson_lambdas = {'TEU': 1.2, 'FEU': 0.6, 'HC': 0.4, 'REEF': 0.2}

    header = ['t', 'arrival_idx', 'max_wtp'] + product_ids

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        total_arrivals = 0
        for t in range(total_hours):
            num_arrivals = np.random.poisson(avg_arrivals_per_hour)
            for arr_idx in range(num_arrivals):
                # Generate bundle
                bundle = {pid: np.random.poisson(poisson_lambdas[pid]) for pid in product_ids}
                # Ensure at least one product is requested
                if all(v == 0 for v in bundle.values()):
                    bundle[random.choice(product_ids)] = 1

                # Calculate bundle's base price
                bundle_base_price = sum(base_prices[pid] * qty for pid, qty in bundle.items())

                # WTP is a random multiplier on the base price (e.g., 1.0x to 1.5x)
                wtp_multiplier = random.uniform(1.0, 1.5)
                max_wtp = bundle_base_price * wtp_multiplier

                row = {'t': t, 'arrival_idx': arr_idx, 'max_wtp': f"{max_wtp:.2f}"}
                row.update(bundle)
                writer.writerow(row)
                total_arrivals += 1

    print(f"Successfully generated scenario with {total_arrivals} arrivals.")


# --- Main execution block ---
if __name__ == '__main__':
    SCENARIO_FILE = "customer_scenario_wtp.csv"

    # 1. Generate the scenario file first (if it doesn't exist)
    if not os.path.exists(SCENARIO_FILE):
        generate_wtp_scenario_file(filepath=SCENARIO_FILE)

    # 2. Define simulation config
    sim_config = {
        "customer_scenario_path": SCENARIO_FILE,
        "num_products": 4,
        "num_price_options_per_product": 3,
        "max_feedback_delay": 3,
        "num_resources": 1,
        "pacing_aggressiveness": 0.5,  # This pacing logic can still be improved
        "use_ts_update": True,
        "use_real_lp": True,
        "verbose": False,  # Set to True for detailed hour-by-hour logs
    }

    # 3. Create simulator and run the evaluation
    simulator = WtpScenarioSimulator(**sim_config)
    results = simulator.run_and_evaluate()

    # 4. Plot the results
    if results:
        simulator.plot_cumulative_revenue(results)