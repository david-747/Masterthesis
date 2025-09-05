import csv
import numpy as np
import os

def generate_normal_rate(filename="normal_rate.csv", total_periods=504, mean_arrivals=10):
    """
    Generates a CSV with a steady stream of customer arrivals.
    The number of arrivals per period follows a Poisson distribution.

    Args:
        filename (str): The name of the output CSV file.
        total_periods (int): The total number of time periods (e.g., hours).
        mean_arrivals (int): The average number of arrivals per period.
    """
    print(f"Generating '{filename}' with mean arrivals: {mean_arrivals}")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['period', 'num_arrivals'])
        for t in range(total_periods):
            # Sample from a Poisson distribution
            arrivals = np.random.poisson(lam=mean_arrivals)
            writer.writerow([t, arrivals])
    print("...Done.")

def generate_demand_shock(filename="demand_shock.csv", total_periods=504, normal_mean=10, shock_period=250, shock_multiplier=10):
    """
    Generates a CSV where most periods have normal demand, but one period
    has a significant spike (a demand shock).

    Args:
        filename (str): The name of the output CSV file.
        total_periods (int): Total time periods.
        normal_mean (int): Average arrivals during normal periods.
        shock_period (int): The period 't' at which the shock occurs.
        shock_multiplier (int): How many times larger the shock is than normal mean.
    """
    print(f"Generating '{filename}' with a shock at period {shock_period}")
    shock_arrivals = normal_mean * shock_multiplier
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['period', 'num_arrivals'])
        for t in range(total_periods):
            if t == shock_period:
                arrivals = np.random.poisson(lam=shock_arrivals)
            else:
                arrivals = np.random.poisson(lam=normal_mean)
            writer.writerow([t, arrivals])
    print("...Done.")

def generate_increasing_demand(filename="increasing_demand.csv", total_periods=504, start_rate=2, end_rate=20):
    """
    Generates a CSV with a customer arrival rate that increases over time.

    Args:
        filename (str): The name of the output CSV file.
        total_periods (int): Total time periods.
        start_rate (int): The initial average arrival rate.
        end_rate (int): The final average arrival rate.
    """
    print(f"Generating '{filename}' with demand increasing from {start_rate} to {end_rate}")
    # Create a linear progression of mean arrival rates
    mean_rates = np.linspace(start_rate, end_rate, total_periods)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['period', 'num_arrivals'])
        for t in range(total_periods):
            arrivals = np.random.poisson(lam=mean_rates[t])
            writer.writerow([t, arrivals])
    print("...Done.")

def generate_decreasing_demand(filename="decreasing_demand.csv", total_periods=504, start_rate=20, end_rate=2):
    """
    Generates a CSV with a customer arrival rate that decreases over time.

    Args:
        filename (str): The name of the output CSV file.
        total_periods (int): Total time periods.
        start_rate (int): The initial average arrival rate.
        end_rate (int): The final average arrival rate.
    """
    print(f"Generating '{filename}' with demand decreasing from {start_rate} to {end_rate}")
    # Create a linear progression of mean arrival rates
    mean_rates = np.linspace(start_rate, end_rate, total_periods)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['period', 'num_arrivals'])
        for t in range(total_periods):
            arrivals = np.random.poisson(lam=mean_rates[t])
            writer.writerow([t, arrivals])
    print("...Done.")


if __name__ == '__main__':
    # --- Configuration ---
    # You can change these parameters to generate different scenarios.
    TOTAL_PERIODS = 504  # 3 weeks * 7 days/week * 24 hours/day

    # Create a directory to store the scenarios if it doesn't exist
    if not os.path.exists('../scenarios'):
        os.makedirs('../scenarios')

    # --- Generate all scenario files ---
    generate_normal_rate(
        filename="normal_rate.csv",
        total_periods=TOTAL_PERIODS,
        mean_arrivals=10
    )

    generate_demand_shock(
        filename="demand_shock.csv",
        total_periods=TOTAL_PERIODS,
        normal_mean=10,
        shock_period=TOTAL_PERIODS // 2, # Shock happens halfway through
        shock_multiplier=15
    )

    generate_increasing_demand(
        filename="increasing_demand.csv",
        total_periods=TOTAL_PERIODS,
        start_rate=2,
        end_rate=25
    )

    generate_decreasing_demand(
        filename="decreasing_demand.csv",
        total_periods=TOTAL_PERIODS,
        start_rate=25,
        end_rate=2
    )

    print("\nAll scenario files have been generated in the 'scenarios/' directory.")