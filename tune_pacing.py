# tune_pacing.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import the simulator class from your saved file
#from simulator import ContextualWtpScenarioSimulator, Season
from ContextualWtpScenarioSimulator import ContextualWtpScenarioSimulator

def run_tuning_experiment():
    """
    Orchestrates the simulation experiment to tune the pacing_aggressiveness parameter.
    """
    print("--- Starting Pacing Aggressiveness Tuning Experiment ---")

    # --- Core Configuration ---
    # The main folder where all results will be stored.
    ROOT_OUTPUT_DIR = "pacing_parameter_tuning_results"

    # Define the range for the pacing_aggressiveness parameter.
    # np.arange is used for floating-point steps.
    PACING_VALUES = np.arange(0.1, 10.1, 0.1)

    # Set the number of times to run the simulation for each parameter value.
    NUM_RUNS_PER_PARAM = 10

    # Specify the scenario file to be used for all simulations.
    SCENARIO_FILE = "scenarios/mid_season_normal.csv"

    # A list to store the summary results from every single run.
    all_run_results = []

    # --- Base Simulation and Pricing Configuration ---
    # This dictionary holds all the settings that will remain constant across runs.
    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    multipliers = {
        0: {'n': 'Aggressive', 'm': 0.85},
        1: {'n': 'Standard', 'm': 1.0},
        2: {'n': 'Premium', 'm': 1.2}
    }

    base_sim_config = {
        "customer_scenario_path": SCENARIO_FILE,
        "num_products": 4,
        "num_price_options_per_product": 3,
        "max_feedback_delay": 0,
        "num_resources": 1,
        "use_ts_update": True,
        "use_contextual_lp": True,
        "verbose": False,  # Set to False to keep the console output clean
        "prior_beliefs_path": None,
        "base_prices": base_prices,
        "multipliers": multipliers
    }

    # Create the main results directory if it doesn't exist.
    os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

    # --- Main Experiment Loop ---
    # Outer loop: Iterates through each value of the pacing parameter.
    for pacing_value in PACING_VALUES:
        # Round to handle potential floating point inaccuracies (e.g., 0.3000000004)
        pacing_value = round(pacing_value, 2)

        # Inner loop: Repeats the simulation for the current pacing value.
        for run_num in range(1, NUM_RUNS_PER_PARAM + 1):

            start_time = datetime.now()
            print(f"Executing: Pacing = {pacing_value:.1f}, Run = {run_num}/{NUM_RUNS_PER_PARAM}...")

            # --- 1. Set up Directories and Paths ---
            pacing_str = f"pacing_{str(pacing_value).replace('.', '_')}"
            run_dir = os.path.join(ROOT_OUTPUT_DIR, pacing_str, f"run_{run_num}")
            os.makedirs(run_dir, exist_ok=True)

            # --- 2. Create Configuration for this Specific Run ---
            # Make a copy of the base config to avoid modifying it.
            current_config = base_sim_config.copy()
            current_config["pacing_aggressiveness"] = pacing_value

            # Set the output paths for this run's log files.
            current_config["metrics_csv_path"] = os.path.join(run_dir, "metrics_log.csv")
            current_config["detailed_log_csv_path"] = os.path.join(run_dir, "detailed_agent_log.csv")

            # Save the configuration file for this run.
            with open(os.path.join(run_dir, "config.json"), 'w') as f:
                json.dump(current_config, f, indent=4)

            # --- 3. Instantiate and Run the Simulator ---
            simulator = ContextualWtpScenarioSimulator(**current_config)
            summary_path = os.path.join(run_dir, "summary.txt")

            # The run_and_evaluate method returns a dictionary of key performance indicators.
            results = simulator.run_and_evaluate(summary_save_path=summary_path)

            # --- 4. Plot and Store Results ---
            if results:
                # Generate and save the revenue plot for this specific run.
                plot_path = os.path.join(run_dir, "revenue_plot.png")
                simulator.plot_cumulative_revenue(results, simulator.metrics_records, save_path=plot_path)

                # Append the key results to our master list.
                all_run_results.append({
                    "pacing_aggressiveness": pacing_value,
                    "run_number": run_num,
                    "achieved_revenue": results["achieved_revenue"],
                    "benchmark_revenue": results["benchmark_revenue"],
                    "performance_percentage": results["performance_percentage"],
                    "regret": results["regret"]
                })

            end_time = datetime.now()
            print(
                f"  > Finished in {(end_time - start_time).total_seconds():.2f} seconds. Revenue: ${results.get('achieved_revenue', 0):,.2f}")

    print("\n--- All simulation runs completed. Aggregating results... ---")

    # --- 5. Aggregate and Summarize All Results ---
    # Convert the list of dictionaries into a pandas DataFrame for easy analysis.
    results_df = pd.DataFrame(all_run_results)

    # Group by the pacing parameter and calculate the mean and standard deviation.
    summary_df = results_df.groupby("pacing_aggressiveness").agg(
        avg_revenue=('achieved_revenue', 'mean'),
        std_revenue=('achieved_revenue', 'std'),
        avg_performance_pct=('performance_percentage', 'mean')
    ).reset_index()

    # Save the aggregated summary to a CSV file in the root output directory.
    summary_csv_path = os.path.join(ROOT_OUTPUT_DIR, "summary_results.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Aggregated summary saved to '{summary_csv_path}'")

    # --- 6. Plot the Final Aggregated Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot the average revenue as a line.
    ax.plot(summary_df["pacing_aggressiveness"], summary_df["avg_revenue"],
            marker='o', linestyle='-', color='b', label="Average Achieved Revenue")

    # Add a shaded area to represent the standard deviation (±1 std dev).
    ax.fill_between(
        summary_df["pacing_aggressiveness"],
        summary_df["avg_revenue"] - summary_df["std_revenue"],
        summary_df["avg_revenue"] + summary_df["std_revenue"],
        color='b', alpha=0.2, label="Standard Deviation"
    )

    # Find and highlight the optimal point.
    optimal_point = summary_df.loc[summary_df['avg_revenue'].idxmax()]
    optimal_pacing = optimal_point['pacing_aggressiveness']
    max_revenue = optimal_point['avg_revenue']

    ax.axvline(x=optimal_pacing, color='r', linestyle='--',
               label=f"Optimal Pacing ≈ {optimal_pacing:.1f} (${max_revenue:,.0f})")

    # Formatting the plot
    ax.set_title("Impact of Pacing Aggressiveness on Revenue", fontsize=16, fontweight='bold')
    ax.set_xlabel("Pacing Aggressiveness Parameter", fontsize=12)
    ax.set_ylabel("Average Achieved Revenue ($)", fontsize=12)
    ax.legend()
    ax.grid(True)

    # Save the final plot.
    final_plot_path = os.path.join(ROOT_OUTPUT_DIR, "pacing_aggressiveness_performance.png")
    plt.savefig(final_plot_path)
    print(f"Final performance plot saved to '{final_plot_path}'")
    plt.show()


# --- Main execution block ---
if __name__ == '__main__':
    run_tuning_experiment()