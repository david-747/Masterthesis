# Improved Contextual WTP Scenario Generator
# Fixes the issues in the original scenario generation script

import csv
import random
import numpy as np
from enum import Enum
from Context import Season, CommodityValue, CustomerType

'''
class Season(Enum):
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


class CustomerType(Enum):
    NEW = "NEW"
    RECURRING = "RECURRING"


class CommodityValue(Enum):
    LOW = "LOW"
    HIGH = "HIGH"
'''

def calculate_contextual_wtp(bundle_base_price, season, customer_type, commodity_value):
    """
    Enhanced WTP calculation that properly uses ALL contextual information.
    """
    # Base multiplier from commodity value
    if commodity_value == CommodityValue.HIGH:
        base_multiplier = random.uniform(1.25, 1.8)
    else:  # LOW commodity
        base_multiplier = random.uniform(0.8, 1.1)

    # Season adjustment - THIS IS THE KEY FIX
    if season == Season.HIGH:
        season_boost = random.uniform(1.1, 1.3)  # 10-30% increase in high season
    elif season == Season.LOW:
        season_boost = random.uniform(0.7, 0.9)  # 10-30% decrease in low season
    else:  # MID season
        season_boost = 1.0

    # Customer type adjustment - ANOTHER KEY FIX
    if customer_type == CustomerType.RECURRING:
        # Recurring customers: mix of loyalty premium and discount expectations
        loyalty_factor = random.uniform(0.95, 1.05)
    else:  # NEW customers
        # New customers: more price-sensitive but variable
        novelty_factor = random.uniform(0.9, 1.1)
        loyalty_factor = novelty_factor

    # Combined WTP calculation
    final_multiplier = base_multiplier * season_boost * loyalty_factor
    return bundle_base_price * final_multiplier


def generate_enhanced_contextual_wtp_scenario(
        filepath="enhanced_customer_scenario_contextual_wtp.csv",
        total_hours=504,
        avg_arrivals_per_hour=3.0,
        season=Season.MID,
        new_customer_ratio=0.5,
        high_value_commodity_ratio=0.3
):
    """
    Enhanced scenario generator that properly implements contextual WTP effects.
    """
    print(f"Generating enhanced contextual customer scenario file at '{filepath}'...")

    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    product_ids = list(base_prices.keys())
    poisson_lambdas = {'TEU': 1.2, 'FEU': 0.6, 'HC': 0.4, 'REEF': 0.2}

    # Adjust arrival patterns based on season
    seasonal_arrival_multiplier = {
        Season.HIGH: 1.3,  # 30% more customers in high season
        Season.MID: 1.0,  # baseline
        Season.LOW: 0.7  # 30% fewer customers in low season
    }

    adjusted_arrivals = avg_arrivals_per_hour * seasonal_arrival_multiplier[season]

    header = ['t', 'arrival_idx', 'max_wtp', 'season', 'customer_type', 'commodity_value'] + product_ids

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        total_arrivals = 0
        for t in range(total_hours):
            num_arrivals = np.random.poisson(adjusted_arrivals)
            for arr_idx in range(num_arrivals):
                # Generate bundle
                bundle = {pid: np.random.poisson(poisson_lambdas[pid]) for pid in product_ids}
                if all(v == 0 for v in bundle.values()):
                    bundle[random.choice(product_ids)] = 1

                # Determine contextual characteristics
                commodity_value = CommodityValue.HIGH if random.random() < high_value_commodity_ratio else CommodityValue.LOW
                customer_type = CustomerType.NEW if random.random() < new_customer_ratio else CustomerType.RECURRING

                # Calculate base bundle price
                bundle_base_price = sum(base_prices[pid] * qty for pid, qty in bundle.items())

                # FIXED WTP CALCULATION - Now uses ALL context
                max_wtp = calculate_contextual_wtp(bundle_base_price, season, customer_type, commodity_value)

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

    print(f"Successfully generated enhanced scenario with {total_arrivals} arrivals.")
    print(f"Season: {season.name}, Expected behavioral differences: IMPLEMENTED")

    # Print expected WTP ranges for validation
    print(f"\nExpected WTP characteristics for {season.name} season:")
    if season == Season.HIGH:
        print("- 10-30% higher WTP across all customer types")
        print("- Higher arrival rate (30% increase)")
    elif season == Season.LOW:
        print("- 10-30% lower WTP across all customer types")
        print("- Lower arrival rate (30% decrease)")
    print("- RECURRING customers: slight loyalty effects (Â±5%)")
    print("- NEW customers: higher price variability (Â±10%)")


# Example usage with CORRECTED parameters
if __name__ == '__main__':
    # Generate LOW season scenario (corrected)
    generate_enhanced_contextual_wtp_scenario(
        filepath="enhanced_low_season_scenario.csv",
        season=Season.LOW,  # FIXED: Now matches filename and creates actual behavioral differences
        new_customer_ratio=0.4,  # Fewer new customers in low season
        high_value_commodity_ratio=0.25,  # Less premium cargo in low season
        avg_arrivals_per_hour=2.0  # Reduced traffic in low season
    )

    # Generate HIGH season scenario for comparison
    generate_enhanced_contextual_wtp_scenario(
        filepath="enhanced_high_season_scenario.csv",
        season=Season.HIGH,
        new_customer_ratio=0.6,  # More new customers in high season
        high_value_commodity_ratio=0.4,  # More premium cargo in high season
        avg_arrivals_per_hour=3.5  # Increased traffic in high season
    )

    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS IMPLEMENTED:")
    print("=" * 60)
    print("âœ“ Season parameter now affects WTP calculation (Â±10-30%)")
    print("âœ“ Customer type creates behavioral differences (Â±5-10%)")
    print("âœ“ Arrival rates vary by season")
    print("âœ“ Commodity mix varies by season")
    print("âœ“ Filename matches actual season behavior")
    print("âœ“ More realistic WTP distributions")