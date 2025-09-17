# visualize_pacing_linked_axis.py

# visualize_pacing_final.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def create_final_plot(csv_path: str, output_image_path: str):
    """
    Loads pacing tuning summary data and creates a final, polished plot.

    This version features:
    - A single plot for revenue, with linked axes for dollars and performance %.
    - A fixed y-axis scale from 0% to 100% on the right.
    - A horizontal line indicating the 100% Oracle Benchmark.
    - Horizontal grid lines corresponding only to the 10% percentage steps.

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

    # --- 1. Calculate the Benchmark (Oracle) Revenue ---
    benchmark_revenue = np.mean(
        summary_df['avg_revenue'] / (summary_df['avg_performance_pct'] / 100)
    )
    print(f"Calculated Oracle Benchmark Revenue: ${benchmark_revenue:,.2f}")

    print("--- Generating final performance plot ---")

    # --- 2. Setup the Plot and Axes ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax2 = ax1.twinx()  # Create secondary axis

    # --- 3. Configure the Secondary Y-Axis (Performance %) FIRST ---
    # We configure this axis first to set the definitive scale.
    ax2.set_ylim(0, 100)  # Fix the scale from 0% to 100%
    ax2.set_ylabel("Performance vs. Oracle (%)", fontsize=12, color='dimgray')
    ax2.tick_params(axis='y', colors='dimgray')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))

    # Add a horizontal line for the 100% benchmark
    ax2.axhline(y=100, color='red', linestyle='--', label=f'Oracle Benchmark (${benchmark_revenue:,.2f})')

    # --- 4. Configure the Primary Y-Axis (Revenue) based on the Secondary Axis ---
    # Link the primary axis scale to the now-fixed secondary axis scale.
    y2_min, y2_max = ax2.get_ylim()
    ax1.set_ylim(
        (y2_min / 100) * benchmark_revenue,
        (y2_max / 100) * benchmark_revenue
    )
    ax1.set_ylabel("Average Achieved Revenue ($)", fontsize=12, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.get_yaxis().set_major_formatter(
        mticker.FuncFormatter(lambda x, p: f'${int(x):,}')
    )

    # --- 5. Plot the Data and Optimal Point ---
    # Plot the main revenue data on the primary axis (ax1)
    ax1.plot(summary_df["pacing_aggressiveness"], summary_df["avg_revenue"],
             marker='o', linestyle='-', color='royalblue', label="Average Achieved Revenue")

    # Add the standard deviation shading
    ax1.fill_between(
        summary_df["pacing_aggressiveness"],
        summary_df["avg_revenue"] - summary_df["std_revenue"],
        summary_df["avg_revenue"] + summary_df["std_revenue"],
        color='royalblue', alpha=0.2
    )

    # Highlight the optimal point
    optimal_point = summary_df.loc[summary_df['avg_revenue'].idxmax()]
    optimal_pacing = optimal_point['pacing_aggressiveness']
    max_revenue = optimal_point['avg_revenue']
    max_percentage = optimal_point['avg_performance_pct']
    ax1.axvline(x=optimal_pacing, color='darkorange', linestyle=':',
                label=f"Optimal Pacing ≈ {optimal_pacing:.2f} (${max_revenue:,.0f}; which is {max_percentage:.2f}% of optimal)")

    # --- 6. Configure Grid Lines ---
    ax1.yaxis.grid(False)  # Turn off horizontal grid for the left axis
    ax1.xaxis.grid(True, linestyle='--')  # Keep vertical grid
    ax2.yaxis.grid(True, linestyle='--')  # Use horizontal grid for the right axis
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))  # Set grid steps to 10%

    # --- 7. Finalize Plot (Title, Legend, and Saving) ---
    fig.suptitle("Impact of Pacing Aggressiveness on Revenue and Performance", fontsize=16, fontweight='bold')

    # Combine legends from both axes for a single, clean legend box
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower center')

    fig.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Success! Final plot saved to '{output_image_path}'")
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    # Specify the path to your results file and the desired output filename.
    INPUT_CSV_FILE = '../pacing_parameter_tuning_low_season_0_25_to_10_better_prices/summary_results.csv'
    OUTPUT_PLOT_FILE = '../pacing_parameter_tuning_low_season_0_25_to_10_better_prices/pacing_performance_dual_axis.png'

    create_final_plot(INPUT_CSV_FILE, OUTPUT_PLOT_FILE)