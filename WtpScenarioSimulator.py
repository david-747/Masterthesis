import csv
import json
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
    ...
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

    def _log_metric(self, row: dict):
        """Append one metric row to in-memory list."""
        self.metrics_records.append(row)

    def _write_metrics_csv(self):
        """Write all collected metric rows to CSV."""
        if not self.metrics_csv_path or not self.metrics_records:
            print("Warning: No metrics to write or CSV path not specified.")
            return
        fieldnames = sorted({k for r in self.metrics_records for k in r.keys()})
        with open(self.metrics_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_records)
        print(f"Metrics successfully saved to '{self.metrics_csv_path}'")

    def _reset_for_new_run(self):
        """Resets the state of the simulator for a fresh run."""
        self.current_inventory = np.copy(self.initial_resource_inventory)
        self.pending_feedback = deque()
        self.current_time_t = 0
        self.metrics_records = []  # Clear metrics for the new run
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

        total_arrivals_in_sim = len(self.arrival_schedule)
        remaining_arrivals_count = total_arrivals_in_sim

        for t in range(self.total_time_periods):
            self.current_time_t = t
            arrivals_this_period = self.arrivals_by_time.get(t, [])
            if self.verbose: print(
                f"--- Agent Hour t = {t}, Inventory: {self.current_inventory}, Arrivals: {len(arrivals_this_period)} ---")

            feedback_to_process = []
            while self.pending_feedback and self.pending_feedback[0][0] <= t:
                feedback_to_process.append(self.pending_feedback.popleft())
            if feedback_to_process:
                self.cmab.process_feedback_for_agent([(fid, s) for _, fid, s in feedback_to_process], t)

            if self.cmab.use_ts_update and remaining_arrivals_count > 0:
                base_budget = np.maximum(self.current_inventory, 0) / remaining_arrivals_count
                resource_constraints = base_budget * self.pacing_aggressiveness
            else:
                resource_constraints = self.initial_resource_inventory / self.total_time_periods

            self.cmab.determine_pricing_policy_for_period(t, resource_constraints)

            for arrival_data in arrivals_this_period:
                observed_context = random.choice(self.all_contexts)
                bundle = arrival_data['bundle']
                max_wtp = arrival_data['max_wtp']

                chosen_pv_id, feedback_map = self.cmab.select_action_and_record_for_feedback(observed_context, t)

                offer_made = chosen_pv_id is not None
                offered_price = 0.0
                if offer_made:
                    price_vector = self.all_price_vectors_map[chosen_pv_id]
                    for pid, qty in bundle.items():
                        if qty > 0:
                            product = next(p for p in self.all_products if p.product_id == pid)
                            offered_price += price_vector.get_price_object(product).amount * qty

                buy = offer_made and (offered_price <= max_wtp)

                revenue_inc = 0.0
                success = False
                inventory_before = self.current_inventory[0]

                if buy:
                    total_required = np.zeros_like(self.current_inventory)
                    for product_id, qty in bundle.items():
                        if qty <= 0: continue
                        idx = self.product_to_idx_map[product_id]
                        total_required += self.resource_consumption_matrix[idx, :] * qty

                    if np.all(self.current_inventory >= total_required):
                        self.current_inventory -= total_required
                        revenue_inc = offered_price
                        total_achieved_revenue += revenue_inc
                        success = True

                self._log_metric({
                    "phase": "agent", "t": t, "arrival_idx": arrival_data['arrival_idx'],
                    "inventory_before": inventory_before,
                    "inventory_after": self.current_inventory[0],
                    "offer_made": int(offer_made), "chosen_pv_id": chosen_pv_id,
                    "offered_price": offered_price, "customer_max_wtp": max_wtp,
                    "accepted": int(success), "revenue_inc": revenue_inc,
                    "pacing_budget": resource_constraints[0]
                })

                if feedback_map is not None:
                    delay = t + random.randint(1, self.max_feedback_delay + 1)
                    for _, feedback_id in feedback_map.items():
                        self.pending_feedback.append((delay, feedback_id, success))

            remaining_arrivals_count -= len(arrivals_this_period)
            self.pending_feedback = deque(sorted(list(self.pending_feedback)))
            cumulative_revenue_over_time.append(total_achieved_revenue)

        return total_achieved_revenue, cumulative_revenue_over_time

    def _run_oracle_simulation(self):
        """Runs the perfect foresight oracle."""
        if self.verbose: print("--- Running Oracle Simulation for Benchmark ---")

        for arrival in self.arrival_schedule:
            required_resources = np.zeros(self.num_resources)
            for pid, qty in arrival['bundle'].items():
                if qty > 0:
                    idx = self.product_to_idx_map[pid]
                    required_resources += self.resource_consumption_matrix[idx, :] * qty
            arrival['resources_needed'] = required_resources

        sorted_arrivals = sorted(self.arrival_schedule, key=lambda x: x['max_wtp'], reverse=True)

        oracle_inventory = np.copy(self.initial_resource_inventory)
        oracle_revenue = 0.0

        for arrival in sorted_arrivals:
            accepted = False
            revenue_inc = 0.0
            inventory_before = oracle_inventory[0]
            if np.all(oracle_inventory >= arrival['resources_needed']):
                oracle_inventory -= arrival['resources_needed']
                oracle_revenue += arrival['max_wtp']
                accepted = True
                revenue_inc = arrival['max_wtp']

            self._log_metric({
                "phase": "oracle", "t": arrival['t'], "arrival_idx": arrival['arrival_idx'],
                "inventory_before": inventory_before, "inventory_after": oracle_inventory[0],
                "offer_made": 1, "offered_price": arrival['max_wtp'], "customer_max_wtp": arrival['max_wtp'],
                "accepted": int(accepted), "revenue_inc": revenue_inc,
            })

        if self.verbose:
            print(f"--- Oracle Simulation Finished. Benchmark Revenue: ${oracle_revenue:,.2f} ---")

        return oracle_revenue, [oracle_revenue] * self.total_time_periods

    # --- THIS METHOD CONTAINS THE PRIMARY FIX ---
    def run_and_evaluate(self, summary_save_path: str | None = None):
        """Calculates benchmark, runs learning agent, and returns performance metrics."""
        # 1. Reset state and run the oracle. The oracle's data is now in self.metrics_records.
        self._reset_for_new_run()
        total_benchmark_revenue, cumulative_benchmark_revenue = self._run_oracle_simulation()

        # 2. IMPORTANT: Save the oracle's data before resetting for the agent.
        oracle_metrics = self.metrics_records.copy()

        # 3. Now, reset the state for a clean agent run.
        self._reset_for_new_run()
        total_achieved_revenue, cumulative_achieved_revenue = self.run()

        # 4. Save the agent's data.
        agent_metrics = self.metrics_records.copy()

        # 5. Combine BOTH sets of data and write the final CSV.
        self.metrics_records = oracle_metrics + agent_metrics
        self._write_metrics_csv()

        regret = total_benchmark_revenue - total_achieved_revenue
        performance_percentage = (
                                         total_achieved_revenue / total_benchmark_revenue) * 100 if total_benchmark_revenue > 0 else 0.0

        # --- NEW: CALCULATE INVENTORY UTILIZATION ---
        initial_inv = self.initial_resource_inventory[0]
        final_inv = self.current_inventory[0]
        inventory_consumed = initial_inv - final_inv
        inventory_utilization_pct = (inventory_consumed / initial_inv) * 100 if initial_inv > 0 else 0.0

        # --- CONSOLE OUTPUT ---
        summary_header = f"\n--- Performance Summary for Scenario: {os.path.basename(self.customer_scenario_path)} ---"
        summary_lines = [
            f"Achieved Revenue: ${total_achieved_revenue:,.2f}",
            f"Benchmark (Oracle) Revenue: ${total_benchmark_revenue:,.2f}",
            f"Regret (Cost of Learning): ${regret:,.2f}",
            f"Percentage of Optimal Revenue Achieved: {performance_percentage:.2f}%",
            f"Inventory Utilization: {inventory_utilization_pct:.2f}% ({int(inventory_consumed):,} of {int(initial_inv):,} units consumed)" # <-- ADDED LINE
        ]

        print(summary_header)
        for line in summary_lines:
            print(line)

        # --- WRITE SUMMARY TO FILE ---
        if summary_save_path:
            try:
                with open(summary_save_path, 'w') as f:
                    f.write(summary_header.strip() + "\n")
                    f.write("\n".join(summary_lines) + "\n")
                print(f"\nSummary report saved to '{summary_save_path}'")
            except Exception as e:
                print(f"Error: Could not write summary to file: {e}")

        return {
            "benchmark_revenue": total_benchmark_revenue, "achieved_revenue": total_achieved_revenue,
            "regret": regret, "performance_percentage": performance_percentage,
            "cumulative_benchmark": cumulative_benchmark_revenue, "cumulative_achieved": cumulative_achieved_revenue
        }

    def plot_cumulative_revenue(self, results, metrics_data, save_path):
        """
        Plots cumulative revenue (left y-axis) and inventory depletion (right y-axis).
        """
        # --- DATA PREPARATION FOR INVENTORY PLOT ---
        agent_metrics = sorted([row for row in metrics_data if row['phase'] == 'agent'], key=lambda x: x['t'])

        inventory_over_time = {}
        if agent_metrics:
            initial_inventory = self.initial_resource_inventory[0]
            inventory_over_time[-1] = initial_inventory
            for row in agent_metrics:
                inventory_over_time[row['t']] = row['inventory_after']

        inv_time_steps = sorted(inventory_over_time.keys())
        inv_levels = [inventory_over_time[t] for t in inv_time_steps]

        final_inv_level = inv_levels[-1] if inv_levels else 0
        initial_inv_level = self.initial_resource_inventory[0]
        remaining_pct = (final_inv_level / initial_inv_level) * 100 if initial_inv_level > 0 else 0
        inventory_label = f'Agent Remaining Inventory ({remaining_pct:.1f}% left)'

        # --- PLOTTING ---
        fig, ax1 = plt.subplots(figsize=(11, 11))

        # AXIS 1: Revenue (Left)
        color_revenue = 'tab:blue'
        ax1.set_xlabel('Time Period (t)', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Cumulative Revenue ($)', color=color_revenue, fontsize=15, fontweight='bold')
        ax1.plot(results['cumulative_achieved'],
                 label=f"Agent Cumulative Revenue (${results['achieved_revenue']:,.0f})", color=color_revenue,
                 linewidth=2)
        ax1.axhline(y=results['benchmark_revenue'], color='red', linestyle='--',
                    label=f"Oracle Max Revenue (${results['benchmark_revenue']:,.0f})")
        # --- NEW: Added labelsize for axis numbers ---
        ax1.tick_params(axis='y', labelcolor=color_revenue, labelsize=15)
        ax1.tick_params(axis='x', labelsize=15)  # For the x-axis numbers
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)

        # AXIS 2: Inventory (Right)
        ax2 = ax1.twinx()
        color_inventory = 'tab:green'
        ax2.set_ylabel('Remaining Inventory (Units)', color=color_inventory, fontsize=15, fontweight='bold')
        ax2.plot(inv_time_steps, inv_levels, label=inventory_label, color=color_inventory, linestyle='--', linewidth=2)
        # --- NEW: Added labelsize for axis numbers ---
        ax2.tick_params(axis='y', labelcolor=color_inventory, labelsize=15)
        ax2.set_ylim(bottom=0)
        ax2.set_ylim(top=initial_inv_level * 1.05)

        '''
        # --- TITLE AND LEGEND ---
        performance_pct = results['performance_percentage']
        title_text = (
            f'Agent Performance: Revenue vs. Inventory Depletion\n'
            f'Scenario: {os.path.basename(self.customer_scenario_path)}\n'
            f'Final Agent Revenue: {performance_pct:.2f}% of Oracle'
        )
        plt.title(title_text, fontsize=16, pad=20)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # --- NEW: Added fontsize for the legend text ---
        ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize=15)

        fig.tight_layout()
        '''

        # --- TITLE AND LEGEND ---
        performance_pct = results['performance_percentage']
        title_text = (
            f'Agent Performance: Revenue vs. Inventory Depletion\n'
            f'Scenario: {os.path.basename(self.customer_scenario_path)}\n'
            f'Final Agent Revenue: {performance_pct:.2f}% of Oracle'
        )
        plt.title(title_text, fontsize=16, pad=20)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # Legend below the plot
        ax2.legend(
            lines + lines2,
            labels + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),  # center below the axes
            fontsize=16,
            ncol=2,  # spread items into 2 columns (adjust as needed)
            frameon=False  # optional: no legend box
        )

        fig.tight_layout(rect=[0, 0.02, 1, 1])  # add bottom margin for legend

        plt.savefig(save_path)
        print(f"\nRevenue and inventory plot saved to '{save_path}'")
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

    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    product_ids = list(base_prices.keys())
    poisson_lambdas = {'TEU': 1.2, 'FEU': 0.6, 'HC': 0.4, 'REEF': 0.2}
    header = ['t', 'arrival_idx', 'max_wtp'] + product_ids

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        total_arrivals = 0
        for t in range(total_hours):
            num_arrivals = np.random.poisson(avg_arrivals_per_hour)
            for arr_idx in range(num_arrivals):
                bundle = {pid: np.random.poisson(poisson_lambdas[pid]) for pid in product_ids}
                if all(v == 0 for v in bundle.values()):
                    bundle[random.choice(product_ids)] = 1

                bundle_base_price = sum(base_prices[pid] * qty for pid, qty in bundle.items())
                wtp_multiplier = random.uniform(1.0, 1.5)
                max_wtp = bundle_base_price * wtp_multiplier

                row = {'t': t, 'arrival_idx': arr_idx, 'max_wtp': f"{max_wtp:.2f}"}
                row.update(bundle)
                writer.writerow(row)
                total_arrivals += 1

    print(f"Successfully generated scenario with {total_arrivals} arrivals.")


# --- Main execution block ---
if __name__ == '__main__':
    output_base_dir = "simulation_outputs"
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(output_base_dir, f"run_{batch_timestamp}")
    os.makedirs(run_folder)

    SCENARIO_FILE = "customer_scenario_wtp.csv"
    if not os.path.exists(SCENARIO_FILE):
        generate_wtp_scenario_file(filepath=SCENARIO_FILE)

    sim_config = {
        "customer_scenario_path": SCENARIO_FILE,
        "num_products": 4,
        "num_price_options_per_product": 3,
        "max_feedback_delay": 3,
        "num_resources": 1,
        "pacing_aggressiveness": 4,
        "use_ts_update": True,
        "use_real_lp": True,
        "verbose": False,
    }

    config_save_path = os.path.join(run_folder, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(sim_config, f, indent=4)
    print(f"Configuration saved to '{config_save_path}'")

    pacing_str = str(sim_config["pacing_aggressiveness"]).replace('.', '_')
    metrics_csv_path = os.path.join(run_folder, f"metrics_log_pacing_{pacing_str}.csv")
    sim_config["metrics_csv_path"] = metrics_csv_path

    simulator = WtpScenarioSimulator(**sim_config)
    #results = simulator.run_and_evaluate()

    # --- NEW: DEFINE SUMMARY PATH AND PASS TO METHOD ---
    summary_txt_path = os.path.join(run_folder, "summary.txt")
    results = simulator.run_and_evaluate(summary_save_path=summary_txt_path)

    if results:
        plot_filename = os.path.join(run_folder, f"revenue_plot_pacing_{pacing_str}.png")
        # --- UPDATED CALL: Pass the simulator's metrics_records to the plot function ---
        simulator.plot_cumulative_revenue(results, simulator.metrics_records, save_path=plot_filename)