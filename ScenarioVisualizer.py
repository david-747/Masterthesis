import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_scenario(ax, file_path, title):
    """
    Loads, processes, and plots the data for a single shipping scenario.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to plot on.
        file_path (str): The path to the CSV file.
        title (str): The title for the subplot.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # --- Data Processing ---
        # Calculate total TEU for each request.
        # FEU (Forty-foot Equivalent Unit) = 2 TEU
        # HC (High Cube) & REEF (Reefer) are assumed to be FEU-sized for volume calc.
        df['total_teu'] = df['TEU'] + (df['FEU'] * 2) + (df['HC'] * 2) + (df['REEF'] * 2)

        # Calculate willingness to pay per TEU, handling requests with 0 TEU to avoid division by zero.
        df['wtp_per_teu'] = df.apply(
            lambda row: row['max_wtp'] / row['total_teu'] if row['total_teu'] > 0 else 0,
            axis=1
        )

        # Group data by hour ('t') for hours that have arrivals
        hourly_agg = df.groupby('t').agg(
            arrival_count=('arrival_idx', 'count'),
            mean_wtp_per_teu=('wtp_per_teu', 'mean')
        )

        # Create a full hourly index from 0 to 504 and reindex the aggregated data.
        # This will create NaNs for hours without arrivals.
        full_index = pd.Index(range(505), name='t')
        hourly_data = hourly_agg.reindex(full_index)

        # Calculate a 24-hour rolling average for WTP. This smooths the line and handles NaNs correctly.
        hourly_data['smooth_wtp_per_teu'] = hourly_data['mean_wtp_per_teu'].rolling(window=24, min_periods=1).mean()

        # Fill NaN arrival counts with 0 for the bar chart
        hourly_data['arrival_count'] = hourly_data['arrival_count'].fillna(0)

        # --- Summary Statistics ---
        total_arrivals = df.shape[0]
        total_teu_sum = df['total_teu'].sum()
        # Calculate overall average WTP per TEU on non-zero TEU requests
        avg_wtp_per_teu_total = df[df['total_teu'] > 0]['wtp_per_teu'].mean()

        # --- Visualization ---
        hours = hourly_data.index

        # Bar plot for arrival rate (zorder=2 makes it appear on top)
        ax.bar(hours, hourly_data['arrival_count'], color='steelblue', label='Request Arrivals', width=1.0, zorder=2,
               alpha=0.75)
        ax.set_xlabel('Hour of Booking Horizon (t)')
        ax.set_ylabel('Number of Requests', color='steelblue')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax.set_ylim(bottom=0)

        # Secondary Y-axis for average WTP
        ax2 = ax.twinx()
        ax2.set_ylabel('Avg. WTP per TEU (€)', color='darkgreen')

        # Line plot for the smoothed average WTP per TEU (zorder=1 makes it appear behind the bars)
        ax2.plot(hours, hourly_data['smooth_wtp_per_teu'], color='darkgreen', label='Avg. WTP/TEU (24h Rolling Avg)',
                 zorder=1)
        ax2.tick_params(axis='y', labelcolor='darkgreen')
        ax2.set_ylim(bottom=0)

        # Adding titles and summary text
        ax.set_title(title, fontsize=12, weight='bold')
        """
        summary_text = (
            f"Total Requests: {total_arrivals}\n"
            f"Total TEU: {total_teu_sum}\n"
            f"Avg. WTP/TEU: €{avg_wtp_per_teu_total:.2f}"
        )
        ax.text(0.02, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
        """

    except FileNotFoundError:
        ax.text(0.5, 0.5, f"Could not find:\n{file_path}", ha='center', va='center', fontsize=10, color='red')
        ax.set_title(title, fontsize=12, weight='bold')


def main():
    """
    Main function to create the visualization dashboard.
    """
    # List of scenario files and their corresponding titles
    scenarios = {
        "Low Demand": "scenarios/low_demand_overcapacity.csv",
        "Mid-Season / Fluctuating WTP": "scenarios/mid_season_fluctuating_wtp.csv",
        "Mid-Season / Increasing WTP": "scenarios/mid_season_increasing_wtp.csv",
        "Peak Season Surge": "scenarios/peak_season_surge.csv"
    }

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle('Demand Simulation Scenarios', fontsize=16, weight='bold')

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each scenario
    for i, (title, file_path) in enumerate(scenarios.items()):
        plot_scenario(axes[i], file_path, title)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make space for suptitle

    # Display the plot
    plt.show()


if __name__ == '__main__':
    main()

