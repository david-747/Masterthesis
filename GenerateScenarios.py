import pandas as pd
import numpy as np
import csv
import random
from Context import Season,CommodityValue,CustomerType

# Define the filename for clarity
original_scenario_file = 'scenarios/customer_scenario_contextual_wtp.csv'
low_season_scenario_file = 'scenarios/low_season_customer_scenario_contextual_wtp.csv'

try:
    # Load your original scenario file
    df = pd.read_csv(original_scenario_file)

    # --- Create the Low-Season Scenario ---

    # 1. Reduce the number of customer arrivals by 30%
    # We sample 70% of the original arrivals to simulate a quieter season
    df_low_season = df.sample(frac=0.7, random_state=42).copy()
    # Sort by the timestamp 't' to maintain a chronological event sequence
    df_low_season.sort_values(by='t', inplace=True)

    # 2. Set the season context to 'LOW' for all entries
    df_low_season['season'] = 'LOW'

    # 3. Reduce the maximum willingness to pay (max_wtp) by 40%
    df_low_season['max_wtp'] = df_low_season['max_wtp'] * 0.75

    # 4. Adjust the product mix to reflect low-season demand
    df_low_season['REEF'] = (df_low_season['REEF'] * 0.2).round().astype(int)
    df_low_season['TEU'] = (df_low_season['TEU'] * 1.1).round().astype(int)
    df_low_season['FEU'] = (df_low_season['FEU'] * 1.1).round().astype(int)
    df_low_season['HC'] = (df_low_season['HC'] * 1.1).round().astype(int)

    # 5. --- NEW: FINAL SAFEGUARD ---
    # After all adjustments, remove any customers with no requested products.
    product_columns = ['TEU', 'FEU', 'HC', 'REEF']

    # Keep only the rows where the total number of products is greater than 0
    customers_before_cleaning = len(df_low_season)
    df_low_season = df_low_season[df_low_season[product_columns].sum(axis=1) > 0]
    customers_after_cleaning = len(df_low_season)

    # Save the new, cleaned scenario to a CSV file
    df_low_season.to_csv(low_season_scenario_file, index=False)

    print(f"Successfully created '{low_season_scenario_file}'")
    print(f"Original number of arrivals: {len(df)}")
    print(f"Arrivals after sampling: {customers_before_cleaning}")
    print(f"Arrivals after cleaning zero-product customers: {customers_after_cleaning}")

except FileNotFoundError:
    print(f"Error: '{original_scenario_file}' not found. Please ensure this file is in the same directory as the script.")


def generate_peak_season_surge_scenario(
        filepath="scenarios/peak_season_surge.csv",
        total_hours=504,
        base_arrivals_per_hour=2.0,
        season=Season.HIGH
):
    """Simulates pre-holiday surge with increasing demand over time."""
    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    product_ids = list(base_prices.keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t', 'arrival_idx', 'max_wtp', 'season',
                                               'customer_type', 'commodity_value'] + product_ids)
        writer.writeheader()

        total_arrivals = 0
        for t in range(total_hours):
            # Demand increases as we approach peak (around t=350)
            surge_multiplier = 1.0 + 2.0 * np.sin(np.pi * t / total_hours) if t < 350 else 0.5
            arrivals_rate = base_arrivals_per_hour * surge_multiplier
            num_arrivals = np.random.poisson(arrivals_rate)

            for arr_idx in range(num_arrivals):
                # More high-value commodities during peak
                high_value_prob = 0.6 if t > 200 and t < 400 else 0.3
                commodity_value = CommodityValue.HIGH if random.random() < high_value_prob else CommodityValue.LOW

                # More new customers during surge
                new_customer_prob = 0.7 if surge_multiplier > 1.5 else 0.4
                customer_type = CustomerType.NEW if random.random() < new_customer_prob else CustomerType.RECURRING

                # Reefer demand increases during peak (perishable holiday goods)
                if t > 250 and t < 400:
                    bundle = {
                        'TEU': np.random.poisson(0.8),
                        'FEU': np.random.poisson(0.5),
                        'HC': np.random.poisson(0.3),
                        'REEF': np.random.poisson(0.6)  # Higher reefer demand
                    }
                else:
                    bundle = {
                        'TEU': np.random.poisson(1.2),
                        'FEU': np.random.poisson(0.6),
                        'HC': np.random.poisson(0.4),
                        'REEF': np.random.poisson(0.2)
                    }

                if all(v == 0 for v in bundle.values()):
                    bundle[random.choice(product_ids)] = 1

                bundle_base_price = sum(base_prices[pid] * qty for pid, qty in bundle.items())

                # WTP increases during peak due to urgency
                if surge_multiplier > 1.5:
                    wtp_boost = 1.1
                else:
                    wtp_boost = 1.0

                if commodity_value == CommodityValue.HIGH:
                    wtp_multiplier = random.uniform(1.25, 1.8) * wtp_boost
                else:
                    wtp_multiplier = random.uniform(0.8, 1.1) * wtp_boost

                max_wtp = bundle_base_price * wtp_multiplier

                writer.writerow({
                    't': t, 'arrival_idx': arr_idx, 'max_wtp': f"{max_wtp:.2f}",
                    'season': season.name,
                    'customer_type': customer_type.name,
                    'commodity_value': commodity_value.name,
                    **bundle
                })
                total_arrivals += 1


def generate_trade_lane_imbalance_scenario(
        filepath="scenarios/trade_lane_imbalance.csv",
        total_hours=504,
        avg_arrivals_per_hour=3.5,
        imbalance_direction="export_heavy"  # or "import_heavy"
):
    """Simulates trade lane imbalances affecting container demand."""
    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    product_ids = list(base_prices.keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t', 'arrival_idx', 'max_wtp', 'season',
                                               'customer_type', 'commodity_value'] + product_ids)
        writer.writeheader()

        total_arrivals = 0
        for t in range(total_hours):
            num_arrivals = np.random.poisson(avg_arrivals_per_hour)

            for arr_idx in range(num_arrivals):
                # Simulate different commodity mixes based on trade direction
                if imbalance_direction == "export_heavy":
                    # Export-heavy: more FEUs, agricultural products need reefers
                    commodity_value = CommodityValue.LOW if random.random() < 0.6 else CommodityValue.HIGH
                    bundle_lambdas = {'TEU': 0.6, 'FEU': 1.2, 'HC': 0.5, 'REEF': 0.4}
                else:  # import_heavy
                    # Import-heavy: more TEUs, consumer goods
                    commodity_value = CommodityValue.HIGH if random.random() < 0.5 else CommodityValue.LOW
                    bundle_lambdas = {'TEU': 1.5, 'FEU': 0.4, 'HC': 0.6, 'REEF': 0.2}

                bundle = {pid: np.random.poisson(lam) for pid, lam in bundle_lambdas.items()}

                if all(v == 0 for v in bundle.values()):
                    bundle[random.choice(product_ids)] = 1

                # Customer mix depends on trade type
                if imbalance_direction == "export_heavy":
                    customer_type = CustomerType.RECURRING if random.random() < 0.6 else CustomerType.NEW
                else:
                    customer_type = CustomerType.NEW if random.random() < 0.7 else CustomerType.RECURRING

                bundle_base_price = sum(base_prices[pid] * qty for pid, qty in bundle.items())

                # WTP affected by scarcity in imbalanced lanes
                scarcity_premium = 1.15 if t > 250 else 1.0

                if commodity_value == CommodityValue.HIGH:
                    wtp_multiplier = random.uniform(1.2, 1.7) * scarcity_premium
                else:
                    wtp_multiplier = random.uniform(0.75, 1.05) * scarcity_premium

                max_wtp = bundle_base_price * wtp_multiplier

                # Season cycles
                season_cycle = t % 168
                if season_cycle < 56:
                    season = Season.LOW
                elif season_cycle < 112:
                    season = Season.MID
                else:
                    season = Season.HIGH

                writer.writerow({
                    't': t, 'arrival_idx': arr_idx, 'max_wtp': f"{max_wtp:.2f}",
                    'season': season.name,
                    'customer_type': customer_type.name,
                    'commodity_value': commodity_value.name,
                    **bundle
                })
                total_arrivals += 1


def generate_disruption_recovery_scenario(
        filepath="scenarios/disruption_recovery.csv",
        total_hours=504,
        disruption_start=100,
        disruption_duration=50
):
    """Simulates demand patterns during and after supply chain disruption."""
    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    product_ids = list(base_prices.keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t', 'arrival_idx', 'max_wtp', 'season',
                                               'customer_type', 'commodity_value'] + product_ids)
        writer.writeheader()

        total_arrivals = 0
        backlog = []  # Accumulated demand during disruption

        for t in range(total_hours):
            if disruption_start <= t < disruption_start + disruption_duration:
                # During disruption: reduced capacity, accumulating backlog
                base_rate = 1.0
                # Still getting inquiries but can't serve many
                for _ in range(np.random.poisson(3.5)):
                    backlog.append(t)
            elif t >= disruption_start + disruption_duration and backlog:
                # Post-disruption: surge from backlog
                base_rate = 5.0
                # Process some backlog
                backlog_to_process = min(len(backlog), np.random.poisson(2))
                backlog = backlog[backlog_to_process:]
            else:
                # Normal operations
                base_rate = 3.0

            num_arrivals = np.random.poisson(base_rate)

            for arr_idx in range(num_arrivals):
                # Post-disruption customers more willing to pay premium
                if t >= disruption_start + disruption_duration and t < disruption_start + disruption_duration + 100:
                    urgency_factor = 1.3
                    commodity_value = CommodityValue.HIGH if random.random() < 0.6 else CommodityValue.LOW
                else:
                    urgency_factor = 1.0
                    commodity_value = CommodityValue.HIGH if random.random() < 0.3 else CommodityValue.LOW

                customer_type = CustomerType.NEW if random.random() < 0.5 else CustomerType.RECURRING

                # Varied bundle sizes post-disruption
                if urgency_factor > 1:
                    bundle = {pid: np.random.poisson(lam * 1.3) for pid, lam in
                              {'TEU': 1.0, 'FEU': 0.8, 'HC': 0.5, 'REEF': 0.3}.items()}
                else:
                    bundle = {pid: np.random.poisson(lam) for pid, lam in
                              {'TEU': 1.2, 'FEU': 0.6, 'HC': 0.4, 'REEF': 0.2}.items()}

                if all(v == 0 for v in bundle.values()):
                    bundle[random.choice(product_ids)] = 1

                bundle_base_price = sum(base_prices[pid] * qty for pid, qty in bundle.items())

                if commodity_value == CommodityValue.HIGH:
                    wtp_multiplier = random.uniform(1.25, 1.8) * urgency_factor
                else:
                    wtp_multiplier = random.uniform(0.8, 1.1) * urgency_factor

                max_wtp = bundle_base_price * wtp_multiplier

                # Season based on actual time
                if t < 168:
                    season = Season.LOW
                elif t < 336:
                    season = Season.MID
                else:
                    season = Season.HIGH

                writer.writerow({
                    't': t, 'arrival_idx': arr_idx, 'max_wtp': f"{max_wtp:.2f}",
                    'season': season.name,
                    'customer_type': customer_type.name,
                    'commodity_value': commodity_value.name,
                    **bundle
                })
                total_arrivals += 1


def generate_low_demand_scenario(
        filepath="scenarios/low_demand_overcapacity.csv",
        total_hours=504,
        base_arrivals_per_hour=1.2,  # Much lower than capacity
        season=Season.LOW
):
    """Simulates overcapacity market with sparse, price-sensitive demand."""
    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    product_ids = list(base_prices.keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t', 'arrival_idx', 'max_wtp', 'season',
                                               'customer_type', 'commodity_value'] + product_ids)
        writer.writeheader()

        total_arrivals = 0
        for t in range(total_hours):
            # Even lower demand during certain periods
            if t % 168 < 24:  # Very low demand at start of each week
                arrivals_rate = base_arrivals_per_hour * 0.5
            elif t % 24 < 8:  # Night hours - minimal demand
                arrivals_rate = base_arrivals_per_hour * 0.3
            else:
                arrivals_rate = base_arrivals_per_hour

            num_arrivals = np.random.poisson(arrivals_rate)

            for arr_idx in range(num_arrivals):
                # In low demand, mostly price-sensitive customers
                commodity_value = CommodityValue.LOW if random.random() < 0.75 else CommodityValue.HIGH

                # Mix of customer types, but more cautious
                customer_type = CustomerType.NEW if random.random() < 0.6 else CustomerType.RECURRING

                # Smaller bundles in weak market
                bundle = {
                    'TEU': np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1]),
                    'FEU': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'HC': np.random.choice([0, 1], p=[0.85, 0.15]),
                    'REEF': np.random.choice([0, 1], p=[0.9, 0.1])
                }

                if all(v == 0 for v in bundle.values()):
                    bundle['TEU'] = 1  # Minimum shipment

                bundle_base_price = sum(base_prices[pid] * qty for pid, qty in bundle.items())

                # Much lower WTP in oversupplied market
                if commodity_value == CommodityValue.HIGH:
                    # Even high-value goods have limited pricing power
                    wtp_multiplier = random.uniform(0.95, 1.25)
                else:
                    # Price-sensitive customers dominate
                    wtp_multiplier = random.uniform(0.65, 0.95)

                # Additional price pressure from competition
                competition_factor = random.uniform(0.9, 1.0)
                max_wtp = bundle_base_price * wtp_multiplier * competition_factor

                writer.writerow({
                    't': t, 'arrival_idx': arr_idx, 'max_wtp': f"{max_wtp:.2f}",
                    'season': season.name,
                    'customer_type': customer_type.name,
                    'commodity_value': commodity_value.name,
                    **bundle
                })
                total_arrivals += 1

        print(
            f"Generated low demand scenario with {total_arrivals} arrivals (avg: {total_arrivals / total_hours:.2f}/hour)")

generate_peak_season_surge_scenario()
generate_low_demand_scenario()
generate_trade_lane_imbalance_scenario()
generate_disruption_recovery_scenario()