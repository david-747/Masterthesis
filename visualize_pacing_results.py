# visualize_pacing_final_simplified.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def create_simplified_final_plot(csv_path: str, output_image_path: str):
    """
    Loads pacing tuning summary data and creates a simplified, final plot.

    This definitive version features:
    - A revenue axis that is guaranteed to start at $0.
    - A simplified revenue axis showing ONLY the Agent Peak and Oracle Benchmark values.
    - Left Y-Axis: Performance vs. Oracle (%).
    - Right Y-Axis: Average Achieved Revenue ($).

    Args:
        csv_path (str): The path to the input summary CSV file.
        output_image_path (str): The path where the output plot image will be saved.
    """
    print(f"--- Loading data from '{csv_path}' ---")
    try:
        summary_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # --- 1. Calculate Benchmark Revenue and other key metrics ---
    benchmark_revenue = np.mean(
        summary_df['avg_revenue'] / (summary_df['avg_performance_pct'] / 100)
    )
    summary_df['std_performance_pct'] = (summary_df['std_revenue'] / benchmark_revenue) * 100

    optimal_point = summary_df.loc[summary_df['avg_revenue'].idxmax()]
    #optimal_pacing = optimal_point['pacing_aggressiveness']
    max_revenue = optimal_point['avg_revenue']
    max_percentage = optimal_point['avg_performance_pct']

    print(f"Calculated Oracle Benchmark Revenue: ${benchmark_revenue:,.2f}")
    print(f"Agent Peak Revenue: ${max_revenue:,.2f}")

    print("--- Generating simplified final performance plot ---")

    # --- 2. Setup Plot and Axes ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax2 = ax1.twinx()

    # --- 3. Configure Axis Labels and Formatting ---
    ax1.set_ylabel("Performance vs. Oracle (%)", fontsize=24, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue', labelsize=18)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(10))

    ax2.set_ylabel("Average Achieved Revenue ($)", fontsize=24, color='dimgray')
    ax2.tick_params(axis='y', colors='dimgray', labelsize=18)
    ax2.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${int(x):,}'))

    # --- 4. Plot Data and Reference Lines on the Primary Axis (ax1) ---
    ax1.plot(summary_df["max_feedback_delay"], summary_df["avg_performance_pct"],
             marker='o', linestyle='-', color='royalblue', label="Agent Performance")
    ax1.fill_between(
        summary_df["max_feedback_delay"],
        summary_df["avg_performance_pct"] - summary_df['std_performance_pct'],
        summary_df["avg_performance_pct"] + summary_df['std_performance_pct'],
        color='royalblue', alpha=0.2
    )
    ax1.axhline(y=100, color='red', linestyle='--', label='Oracle Benchmark')
    ax1.axhline(y=max_percentage, color='green', linestyle=':', label=f'Agent Peak Performance: {max_percentage:,.2f}% of oracle')
    #ax1.axvline(x=optimal_pacing, color='darkorange', linestyle=':', label=f"Optimal Pacing ≈ {optimal_pacing:.2f}")

    # --- 5. Set Ticks and Finalize Axis Limits (The Correct Way) ---
    # First, set the revenue ticks on the right axis to ONLY the two key values.
    ax2.set_yticks([max_revenue, benchmark_revenue])

    # Second, enforce the final limits for the revenue axis (right).
    # This starts at 0 and adds 5% padding at the top.
    ax2.set_ylim(0, benchmark_revenue * 1.05)

    # Finally, sync the percentage axis (left) to the new, definitive revenue axis limits.
    y2_min, y2_max = ax2.get_ylim()
    ax1.set_ylim((y2_min / benchmark_revenue) * 100, (y2_max / benchmark_revenue) * 100)

    # --- 6. Configure Grid Lines ---
    ax1.set_ylim(50, 105)
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(True, linestyle='--')
    ax2.yaxis.grid(False)

    # --- 7. Finalize Plot ---
    fig.suptitle("Impact of Delay on Performance (mid season fluctuating WTP)", fontsize=24, fontweight='bold')
    ax1.legend(loc='lower center', fontsize=20)
    fig.tight_layout()

    plt.savefig(output_image_path, dpi=300)
    print(f"Success! Final plot saved to '{output_image_path}'")
    plt.show()


# --- Main execution block ---
if __name__ == '__main__':
    INPUT_CSV_FILE = 'delay_parameter_tuning_results/summary_delay_results.csv'
    OUTPUT_PLOT_FILE = 'delay_parameter_tuning_results/delay_performance.png'

    create_simplified_final_plot(INPUT_CSV_FILE, OUTPUT_PLOT_FILE)