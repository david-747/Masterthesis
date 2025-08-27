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
from LPSolver import solve_real_lp


class ScenarioSimulator:
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
                 bundle_file_path: str | None = None,
                 poisson_lambdas: dict[str, float] | None = None,
                 metrics_csv_path: str | None = "metrics_log.csv"
                 ):
        print("Initializing ScenarioSimulator...")

        # --- Store configurations ---
        self.customer_scenario_path = customer_scenario_path
        self.pacing_aggressiveness = pacing_aggressiveness
        self.use_ts_update = use_ts_update
        self.verbose = verbose
        self.bundle_file_path = bundle_file_path
        self.poisson_lambdas = poisson_lambdas if poisson_lambdas is not None else {'TEU': 1.2, 'FEU': 0.6, 'HC': 0.4, 'REEF': 0.2}
        self.metrics_csv_path = metrics_csv_path
        self.metrics_records: list[dict] = []

        self.arrival_schedule = self._load_customer_scenario(customer_scenario_path)
        if not self.arrival_schedule:
            raise ValueError("Customer scenario could not be loaded or is empty.")
        self.total_time_periods = len(self.arrival_schedule)
        print(
            f"Loaded scenario '{os.path.basename(customer_scenario_path)}' with {self.total_time_periods} periods and {sum(self.arrival_schedule)} total arrivals.")

        self.num_products = num_products
        self.num_price_options = num_price_options_per_product
        self.max_feedback_delay = max_feedback_delay
        self.num_resources = num_resources

        # --- Common components ---
        self.all_products = self._create_products()
        self.all_price_vectors_map, self.all_price_indices = self._create_price_vectors(self.all_products)
        self.all_contexts = self._create_contexts()
        self.true_demand_theta = self._create_ground_truth_demand()
        self.resource_consumption_matrix, self.initial_resource_inventory = self._initialize_resources()
        self.product_to_idx_map = {product.product_id: i for i, product in enumerate(self.all_products)}
        self.solver_function = solve_real_lp if use_real_lp else None
        self.demand_scaling_factor = 1.0
        # Generate or load fixed customer bundles (quantities per arrival)
        self.customer_bundles = self._load_or_generate_bundles()

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

    def _load_customer_scenario(self, filepath: str) -> list[int]:
        schedule = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    schedule.append(int(row['num_arrivals']))
            return schedule
        except FileNotFoundError:
            return []
        except Exception:
            return []

    def _load_or_generate_bundles(self) -> list[list[dict[str, int]]]:
        """
        Returns a list of length T (time periods); each element is a list (len = arrivals at t)
        of dicts mapping product_id -> quantity demanded by that arrival.
        If bundle_file_path is provided and exists, we load from CSV; otherwise we generate and (if path given) save.
        CSV format:
        t,arrival_idx,TEU,FEU,HC,REEF
        """
        bundles: list[list[dict[str, int]]] = []
        # If we have a CSV path and it exists, load it
        if self.bundle_file_path and os.path.exists(self.bundle_file_path):
            with open(self.bundle_file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                # Prepare empty structure
                for _ in range(self.total_time_periods):
                    bundles.append([])
                for row in reader:
                    t = int(row['t'])
                    arrival_idx = int(row['arrival_idx'])
                    q = {pid: int(row.get(pid, 0)) for pid in [p.product_id for p in self.all_products]}
                    # Ensure list length
                    while len(bundles[t]) <= arrival_idx:
                        bundles[t].append({})
                    bundles[t][arrival_idx] = q
            return bundles

        # Otherwise generate
        for t in range(self.total_time_periods):
            period_bundles = []
            for arrival_idx in range(self.arrival_schedule[t]):
                q_dict: dict[str, int] = {}
                # Draw Poisson quantity per product_id
                for p in self.all_products:
                    lam = self.poisson_lambdas.get(p.product_id, 0.0)
                    qty = np.random.poisson(lam)
                    q_dict[p.product_id] = int(qty)
                # If all quantities are zero, force at least one unit of a random product to keep demand > 0
                if all(v == 0 for v in q_dict.values()):
                    rnd_prod = random.choice(self.all_products).product_id
                    q_dict[rnd_prod] = 1
                period_bundles.append(q_dict)
            bundles.append(period_bundles)

        # Save if path was provided
        if self.bundle_file_path:
            header = ['t', 'arrival_idx'] + [p.product_id for p in self.all_products]
            file_exists = os.path.isfile(self.bundle_file_path)
            with open(self.bundle_file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for t, pb in enumerate(bundles):
                    for idx, q in enumerate(pb):
                        row = {'t': t, 'arrival_idx': idx}
                        row.update(q)
                        writer.writerow(row)
        return bundles

    def _simulate_demand(self, product: Product, chosen_price_vector_id: int, context: Context) -> bool:
        if chosen_price_vector_id is None: return False
        true_prob = self.true_demand_theta[context][product][chosen_price_vector_id]
        return random.random() < true_prob

    def _purchase_probability(self, context: Context, chosen_price_vector_id: int, bundle: dict[str, int]) -> float:
        """Average conversion probability over products with positive quantity."""
        active_products = [p for p in self.all_products if bundle.get(p.product_id, 0) > 0]
        if not active_products:
            return 0.0
        probs = [self.true_demand_theta[context][p][chosen_price_vector_id] for p in active_products]
        return float(sum(probs) / len(probs))

    def _log_metric(self, row: dict):
        """Append one metric row to in-memory list."""
        self.metrics_records.append(row)

    def _write_metrics_csv(self):
        """Write all collected metric rows to CSV."""
        if not self.metrics_csv_path or not self.metrics_records:
            return
        fieldnames = sorted({k for r in self.metrics_records for k in r.keys()})
        file_exists = os.path.isfile(self.metrics_csv_path)
        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in self.metrics_records:
                writer.writerow(r)

    def run(self):
        """Runs the simulation with the learning agent, returns revenue and cumulative revenue list."""
        if self.verbose: print(f"\n--- Running Learning Agent Simulation ---")
        total_achieved_revenue = 0
        cumulative_revenue_over_time = []

        for t in range(self.total_time_periods):
            self.current_time_t = t
            if self.verbose: print(
                f"--- Agent Hour t = {t}, Inventory: {self.current_inventory}, Arrivals: {self.arrival_schedule[t]} ---")

            feedback_to_process = []
            while self.pending_feedback and self.pending_feedback[0][0] <= t:
                feedback_to_process.append(self.pending_feedback.popleft())

            if feedback_to_process:
                self.cmab.process_feedback_for_agent([(fid, s) for _, fid, s in feedback_to_process], t)

            remaining_arrivals = sum(self.arrival_schedule[t:])
            if self.cmab.use_ts_update and remaining_arrivals > 0:
                base_budget = np.maximum(self.current_inventory, 0) / remaining_arrivals
                resource_constraints = base_budget * self.pacing_aggressiveness
            else:
                resource_constraints = self.initial_resource_inventory / self.total_time_periods

            self.cmab.determine_pricing_policy_for_period(t, resource_constraints)

            for arrival_idx in range(self.arrival_schedule[t]):
                observed_context = random.choice(self.all_contexts)
                bundle = self.customer_bundles[t][arrival_idx]
                chosen_pv_id, feedback_map = self.cmab.select_action_and_record_for_feedback(observed_context, t)

                offer_made = int(chosen_pv_id is not None)
                if not offer_made:
                    # Log a no-offer row
                    self._log_metric({
                        "phase": "agent",
                        "t": t,
                        "arrival_idx": arrival_idx,
                        "offer_made": 0,
                        "accepted": 0,
                        "revenue_inc": 0.0,
                        "inventory_before": float(self.current_inventory[0]),
                        "inventory_after": float(self.current_inventory[0]),
                        "remaining_arrivals": int(sum(self.arrival_schedule[t:])),
                        "resource_constraint": float(resource_constraints[0]),
                        "chosen_pv_id": None,
                        "avg_conv_prob": 0.0,
                        "qty_requested": sum(bundle.values()),
                        "qty_sold": 0
                    })
                    continue

                inv_before = float(self.current_inventory[0])
                chosen_pv_obj = self.all_price_vectors_map[chosen_pv_id]
                p_conv = self._purchase_probability(observed_context, chosen_pv_id, bundle)
                buy = random.random() < p_conv

                revenue_inc = 0.0
                success = False
                qty_sold = 0
                if buy:
                    total_required = np.zeros_like(self.current_inventory)
                    for product_id, qty in bundle.items():
                        if qty <= 0: continue
                        idx = self.product_to_idx_map[product_id]
                        total_required += self.resource_consumption_matrix[idx, :] * qty
                    if np.all(self.current_inventory >= total_required):
                        self.current_inventory -= total_required
                        revenue_inc = sum(
                            self.all_price_vectors_map[chosen_pv_id].get_price_object(
                                next(p for p in self.all_products if p.product_id == pid)
                            ).amount * qty
                            for pid, qty in bundle.items() if qty > 0
                        )
                        total_achieved_revenue += revenue_inc
                        success = True
                        qty_sold = sum(bundle.values())

                inv_after = float(self.current_inventory[0])

                # Record same success for all feedback_ids (one per product in feedback_map)
                for _, feedback_id in feedback_map.items():
                    self.pending_feedback.append(
                        (t + random.randint(1, self.max_feedback_delay + 1), feedback_id, success))

                self._log_metric({
                    "phase": "agent",
                    "t": t,
                    "arrival_idx": arrival_idx,
                    "offer_made": offer_made,
                    "accepted": int(success),
                    "revenue_inc": float(revenue_inc),
                    "inventory_before": inv_before,
                    "inventory_after": inv_after,
                    "remaining_arrivals": int(sum(self.arrival_schedule[t:])),
                    "resource_constraint": float(resource_constraints[0]),
                    "chosen_pv_id": chosen_pv_id,
                    "avg_conv_prob": float(p_conv),
                    "qty_requested": sum(bundle.values()),
                    "qty_sold": qty_sold
                })

            self.pending_feedback = deque(sorted(list(self.pending_feedback)))
            cumulative_revenue_over_time.append(total_achieved_revenue)

        return total_achieved_revenue, cumulative_revenue_over_time

    def _run_oracle_simulation(self):
        """Runs an optimized simulation with a 'perfect' agent."""
        if self.verbose: print("--- Running Oracle Simulation for Benchmark ---")
        oracle_revenue = 0
        oracle_inventory = np.copy(self.initial_resource_inventory)
        cumulative_revenue_over_time = []

        for t in range(self.total_time_periods):
            if self.verbose: print(
                f"--- Oracle Hour t = {t}, Inventory: {oracle_inventory}, Arrivals: {self.arrival_schedule[t]} ---")

            remaining_arrivals = sum(self.arrival_schedule[t:])
            if self.use_ts_update and remaining_arrivals > 0:
                base_budget = np.maximum(oracle_inventory, 0) / remaining_arrivals
                resource_constraints = base_budget * self.pacing_aggressiveness
            else:
                resource_constraints = self.initial_resource_inventory / self.total_time_periods

            oracle_lp_solution = solve_real_lp(
                sampled_theta_t=self.true_demand_theta,
                resource_constraints_c_j=resource_constraints,
                all_contexts=self.all_contexts, all_products=self.all_products,
                all_price_indices=self.all_price_indices, all_price_vectors_map=self.all_price_vectors_map,
                resource_consumption_matrix_A_ij=self.resource_consumption_matrix,
                demand_scaling_factor=1.0, context_probabilities=None, product_to_idx_map=self.product_to_idx_map
            )

            for arrival_idx in range(self.arrival_schedule[t]):
                observed_context = random.choice(self.all_contexts)
                bundle = self.customer_bundles[t][arrival_idx]
                probs_for_context = oracle_lp_solution[observed_context]

                price_indices = list(probs_for_context.keys())
                weights = list(probs_for_context.values())
                p_none = max(0.0, 1.0 - sum(weights))
                choices = price_indices + [None]
                weights = weights + [p_none]
                chosen_pv_id = random.choices(choices, weights=weights, k=1)[0]

                offer_made = int(chosen_pv_id is not None)
                inv_before = float(oracle_inventory[0])

                if not offer_made:
                    self._log_metric({
                        "phase": "oracle",
                        "t": t,
                        "arrival_idx": arrival_idx,
                        "offer_made": 0,
                        "accepted": 0,
                        "revenue_inc": 0.0,
                        "inventory_before": inv_before,
                        "inventory_after": inv_before,
                        "remaining_arrivals": int(sum(self.arrival_schedule[t:])),
                        "resource_constraint": float(resource_constraints[0]),
                        "chosen_pv_id": None,
                        "avg_conv_prob": 0.0,
                        "qty_requested": sum(bundle.values()),
                        "qty_sold": 0
                    })
                    continue

                chosen_pv_obj = self.all_price_vectors_map[chosen_pv_id]
                p_conv = self._purchase_probability(observed_context, chosen_pv_id, bundle)
                buy = random.random() < p_conv

                revenue_before = oracle_revenue
                qty_sold = 0
                if buy:
                    total_required = np.zeros_like(oracle_inventory)
                    for product_id, qty in bundle.items():
                        if qty <= 0: continue
                        idx = self.product_to_idx_map[product_id]
                        total_required += self.resource_consumption_matrix[idx, :] * qty
                    if np.all(oracle_inventory >= total_required):
                        oracle_inventory -= total_required
                        oracle_revenue += sum(
                            self.all_price_vectors_map[chosen_pv_id].get_price_object(
                                next(p for p in self.all_products if p.product_id == pid)
                            ).amount * qty
                            for pid, qty in bundle.items() if qty > 0
                        )
                        qty_sold = sum(bundle.values())

                inv_after = float(oracle_inventory[0])
                revenue_inc = oracle_revenue - revenue_before

                self._log_metric({
                    "phase": "oracle",
                    "t": t,
                    "arrival_idx": arrival_idx,
                    "offer_made": offer_made,
                    "accepted": int(revenue_inc > 0),
                    "revenue_inc": float(revenue_inc),
                    "inventory_before": inv_before,
                    "inventory_after": inv_after,
                    "remaining_arrivals": int(sum(self.arrival_schedule[t:])),
                    "resource_constraint": float(resource_constraints[0]),
                    "chosen_pv_id": chosen_pv_id,
                    "avg_conv_prob": float(p_conv),
                    "pv_weight": float(probs_for_context.get(chosen_pv_id, 0.0)),
                    "qty_requested": sum(bundle.values()),
                    "qty_sold": qty_sold
                })

            cumulative_revenue_over_time.append(oracle_revenue)

        if self.verbose: print(f"--- Oracle Simulation Finished. Benchmark Revenue: ${oracle_revenue:,.2f} ---")
        return oracle_revenue, cumulative_revenue_over_time

    def run_and_evaluate(self):
        """Calculates benchmark, runs learning agent, and returns performance metrics."""
        self._reset_for_new_run()
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

        # Write metrics to CSV
        self._write_metrics_csv()

        return {
            "benchmark_revenue": total_benchmark_revenue, "achieved_revenue": total_achieved_revenue,
            "regret": regret, "performance_percentage": performance_percentage,
            "cumulative_benchmark": cumulative_benchmark_revenue, "cumulative_achieved": cumulative_achieved_revenue
        }

    def plot_cumulative_regret(self, results):
        """Plots the cumulative regret over time and saves it to a file."""
        cumulative_benchmark = np.array(results['cumulative_benchmark'])
        cumulative_achieved = np.array(results['cumulative_achieved'])
        cumulative_regret = cumulative_benchmark - cumulative_achieved

        plt.figure(figsize=(12, 7))
        plt.plot(cumulative_regret)
        plt.title(f'Cumulative Regret Over Time\nScenario: {os.path.basename(self.customer_scenario_path)}')
        plt.xlabel('Time Period (t)')
        plt.ylabel('Cumulative Regret ($)')
        plt.grid(True, linestyle='--', alpha=0.6)

        plot_filename = f"regret_plot_{os.path.basename(self.customer_scenario_path).replace('.csv', '')}.png"
        plt.savefig(plot_filename)
        print(f"\nRegret plot saved to '{plot_filename}'")
        plt.close()

    # --- HELPER METHODS ---
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

    def _create_ground_truth_demand(self):
        theta = {}
        for ctx in self.all_contexts:
            theta[ctx] = {}
            for p in self.all_products:
                theta[ctx][p] = {}
                for pv_id in self.all_price_indices:
                    prob = 0.7 - (0.2 * pv_id) + (0.1 if "HIGH" in str(ctx.season) else 0)
                    theta[ctx][p][pv_id] = max(0.05, min(0.95, prob))
        return theta


# --- Main execution block ---
if __name__ == '__main__':
    # For a quick, detailed run with a plot:
    print("--- Running a single detailed simulation with regret plot ---")

    # 1. Define simulation config
    single_run_config = {
        "customer_scenario_path": "scenarios/normal_rate.csv",
        "num_products": 4, "num_price_options_per_product": 3,
        "max_feedback_delay": 3, "num_resources": 1,
        "pacing_aggressiveness": 0.5, "use_ts_update": True,
        "use_real_lp": True, "verbose": True,
        "bundle_file_path": "scenarios/fixed_bundles_normal_rate.csv",
        "poisson_lambdas": {"TEU": 1.2, "FEU": 0.6, "HC": 0.4, "REEF": 0.2},
        "metrics_csv_path": "metrics_log_normal_rate.csv",
    }

    # 2. Create simulator and run the evaluation
    simulator = ScenarioSimulator(**single_run_config)
    results = simulator.run_and_evaluate()

    # 3. Plot the results
    if results:
        simulator.plot_cumulative_regret(results)

    # To run the full batch experiment and log to CSV (as before):
    # print("\n\n--- Starting batch experiment ---")
    # output_filename = "scenario_simulation_results.csv"
    # scenarios_to_run = ["scenarios/normal_rate.csv", "scenarios/demand_shock.csv", "scenarios/increasing_demand.csv", "scenarios/decreasing_demand.csv"]
    # num_runs_per_scenario = 5
    # batch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # fieldnames = ['timestamp', 'scenario_path', 'run_number', 'pacing_aggressiveness', 'use_ts_update', 'benchmark_revenue', 'achieved_revenue', 'regret', 'performance_percentage']
    # file_exists = os.path.isfile(output_filename)
    # with open(output_filename, 'a', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     if not file_exists: writer.writeheader()
    #     for scenario_path in scenarios_to_run:
    #         if not os.path.exists(scenario_path): continue
    #         for i in range(num_runs_per_scenario):
    #             print(f"\n--- Starting Run #{i + 1}/{num_runs_per_scenario} for Scenario: {os.path.basename(scenario_path)} ---")
    #             sim_config = {"customer_scenario_path": scenario_path, "num_products": 4, "num_price_options_per_product": 3, "max_feedback_delay": 3, "num_resources": 1, "pacing_aggressiveness": 0.75, "use_ts_update": True, "use_real_lp": True, "verbose": False}
    #             simulator = ScenarioSimulator(**sim_config)
    #             results = simulator.run_and_evaluate()
    #             if results:
    #                 log_data = {'timestamp': batch_timestamp, 'scenario_path': os.path.basename(scenario_path), 'run_number': i + 1, 'pacing_aggressiveness': sim_config['pacing_aggressiveness'], 'use_ts_update': sim_config['use_ts_update'], **results}
    #                 del log_data['cumulative_benchmark'] # Don't need these lists in the CSV
    #                 del log_data['cumulative_achieved']
    #                 writer.writerow(log_data)
    #                 csvfile.flush()
    # print(f"\n--- All batch runs complete. Results saved to {output_filename} ---")
