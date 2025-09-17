# Simple Single-Context Scenario Generator (Robust Version)
# Addresses the exploration paralysis by reducing complexity

import csv
import random
import numpy as np
import os
from enum import Enum


class Season(Enum):
    HIGH = "HIGH"


class CustomerType(Enum):
    NEW = "NEW"


class CommodityValue(Enum):
    HIGH = "HIGH"


def generate_simple_scenario(
        filepath="simple_single_context_scenario.csv",
        total_hours=504,
        avg_arrivals_per_hour=3.0,
        season=Season.HIGH,
        customer_type=CustomerType.NEW,
        commodity_value=CommodityValue.HIGH
):
    """
    Generates a single-context scenario to test agent learning without complexity.
    All customers have the same context - only product bundles and WTP vary.
    """

    # Get absolute path and check working directory
    abs_filepath = os.path.abspath(filepath)
    current_dir = os.getcwd()

    print(f"Current working directory: {current_dir}")
    print(f"Attempting to create file at: {abs_filepath}")
    print(f"Directory exists: {os.path.exists(os.path.dirname(abs_filepath))}")

    # Check if we can write to the directory
    directory = os.path.dirname(abs_filepath)
    if not os.access(directory, os.W_OK):
        print(f"ERROR: No write permission to directory: {directory}")
        return

    print(f"Generating simple single-context scenario...")

    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    product_ids = list(base_prices.keys())
    poisson_lambdas = {'TEU': 1.2, 'FEU': 0.6, 'HC': 0.4, 'REEF': 0.2}

    header = ['t', 'arrival_idx', 'max_wtp', 'season', 'customer_type', 'commodity_value'] + product_ids

    try:
        with open(abs_filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            total_arrivals = 0
            for t in range(total_hours):
                num_arrivals = np.random.poisson(avg_arrivals_per_hour)
                for arr_idx in range(num_arrivals):
                    # Generate bundle
                    bundle = {pid: np.random.poisson(poisson_lambdas[pid]) for pid in product_ids}
                    if all(v == 0 for v in bundle.values()):
                        bundle[random.choice(product_ids)] = 1

                    # Calculate base bundle price
                    bundle_base_price = sum(base_prices[pid] * qty for pid, qty in bundle.items())

                    # Single context WTP - HIGH season, NEW customer, HIGH commodity
                    # These customers should be willing to pay premium (1.2-1.6x base price)
                    wtp_multiplier = random.uniform(1.2, 1.6)
                    max_wtp = bundle_base_price * wtp_multiplier

                    row = {
                        't': t,
                        'arrival_idx': arr_idx,
                        'max_wtp': f"{max_wtp:.2f}",
                        'season': season.name,
                        'customer_type': customer_type.name,
                        'commodity_value': commodity_value.name
                    }
                    row.update(bundle)
                    writer.writerow(row)
                    total_arrivals += 1

        print(f"âœ… Successfully generated simple scenario with {total_arrivals} arrivals.")
        print(f"âœ… File saved to: {abs_filepath}")
        print(f"All customers: {season.name} season, {customer_type.name}, {commodity_value.name} commodity")
        print(f"WTP range: 1.2-1.6x base price (should accept pricing levels 1-2)")

        # Calculate expected acceptance rates for different pricing levels
        print(f"\nExpected acceptance rates:")
        print(f"- Level 0 (1.0x base): ~100% (all customers can afford)")
        print(f"- Level 1 (1.2x base): ~80% (customers with WTP > 1.2x)")
        print(f"- Level 2 (1.5x base): ~40% (customers with WTP > 1.5x)")
        print(f"\nAgent should learn these patterns quickly with single context!")

    except Exception as e:
        print(f"ERROR: Failed to create file: {e}")
        print(f"Trying alternative location...")

        # Fallback: try creating in current directory
        fallback_path = "simple_single_context_scenario.csv"
        try:
            with open(fallback_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                # Same generation logic...
                total_arrivals = 0
                for t in range(total_hours):
                    num_arrivals = np.random.poisson(avg_arrivals_per_hour)
                    for arr_idx in range(num_arrivals):
                        bundle = {pid: np.random.poisson(poisson_lambdas[pid]) for pid in product_ids}
                        if all(v == 0 for v in bundle.values()):
                            bundle[random.choice(product_ids)] = 1
                        bundle_base_price = sum(base_prices[pid] * qty for pid, qty in bundle.items())
                        wtp_multiplier = random.uniform(1.2, 1.6)
                        max_wtp = bundle_base_price * wtp_multiplier
                        row = {
                            't': t, 'arrival_idx': arr_idx, 'max_wtp': f"{max_wtp:.2f}",
                            'season': season.name, 'customer_type': customer_type.name,
                            'commodity_value': commodity_value.name
                        }
                        row.update(bundle)
                        writer.writerow(row)
                        total_arrivals += 1

            print(f"âœ… Fallback successful! File created at: {os.path.abspath(fallback_path)}")
            print(f"âœ… Generated {total_arrivals} arrivals")

        except Exception as e2:
            print(f"ERROR: Both attempts failed: {e2}")


# Example usage
if __name__ == '__main__':

    # List current directory contents to debug
    print("Current directory contents:")
    for item in os.listdir('../simulation_outputs_contextual/run_2025-08-29_09-20-29'):
        print(f"  {item}")

    print(f"\nLooking for scenarios directory...")
    if os.path.exists("../scenarios"):
        print("âœ… scenarios directory found")
        scenarios_contents = os.listdir("../scenarios")
        print("scenarios directory contents:")
        for item in scenarios_contents:
            print(f"  {item}")
    else:
        print("âŒ scenarios directory not found")

    print(f"\n" + "=" * 60)

    generate_simple_scenario(
        filepath="simple_single_context_scenario.csv",
        season=Season.HIGH,
        customer_type=CustomerType.NEW,
        commodity_value=CommodityValue.HIGH,
        avg_arrivals_per_hour=3.0
    )

    print("\n" + "=" * 60)
    print("TESTING RECOMMENDATIONS:")
    print("=" * 60)
    print("1. Run agent on this simple scenario first")
    print("2. Should achieve >80% performance vs oracle")
    print("3. If successful, gradually add complexity")
    print("\nEXPECTED LEARNING SPEED:")
    print("- Only 12 parameters to learn (4 products Ã— 3 price levels)")
    print("- Should converge within 100-200 customer interactions")