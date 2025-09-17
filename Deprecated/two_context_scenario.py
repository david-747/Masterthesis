# Two-Context Scenario Generator
# Next step: Add one more context to build agent confidence gradually

import csv
import random
import numpy as np
import os
from enum import Enum


class Season(Enum):
    HIGH = "HIGH"


class CustomerType(Enum):
    NEW = "NEW"
    RECURRING = "RECURRING"  # Add recurring customers


class CommodityValue(Enum):
    HIGH = "HIGH"


def generate_two_context_scenario(
        filepath="two_context_scenario.csv",
        total_hours=504,
        avg_arrivals_per_hour=3.0
):
    """
    Two contexts: HIGH-NEW-HIGH and HIGH-RECURRING-HIGH
    This tests if agent can learn to differentiate between customer types
    """
    print(f"Generating two-context scenario at '{filepath}'...")

    base_prices = {'TEU': 2500.0, 'FEU': 4500.0, 'HC': 4800.0, 'REEF': 8000.0}
    product_ids = list(base_prices.keys())
    poisson_lambdas = {'TEU': 1.2, 'FEU': 0.6, 'HC': 0.4, 'REEF': 0.2}

    header = ['t', 'arrival_idx', 'max_wtp', 'season', 'customer_type', 'commodity_value'] + product_ids

    with open(filepath, 'w', newline='') as f:
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

                # Two contexts with different WTP patterns:
                if random.random() < 0.7:  # 70% NEW customers
                    customer_type = CustomerType.NEW
                    # NEW customers: premium pricing (1.2-1.6x) - same as before
                    wtp_multiplier = random.uniform(1.2, 1.6)
                else:  # 30% RECURRING customers
                    customer_type = CustomerType.RECURRING
                    # RECURRING customers: even higher WTP (1.4-1.8x) - more price tolerance
                    wtp_multiplier = random.uniform(1.4, 1.8)

                max_wtp = bundle_base_price * wtp_multiplier

                row = {
                    't': t,
                    'arrival_idx': arr_idx,
                    'max_wtp': f"{max_wtp:.2f}",
                    'season': Season.HIGH.name,
                    'customer_type': customer_type.name,
                    'commodity_value': CommodityValue.HIGH.name
                }
                row.update(bundle)
                writer.writerow(row)
                total_arrivals += 1

    print(f"âœ… Successfully generated two-context scenario with {total_arrivals} arrivals.")
    print(f"ðŸ“Š Context distribution:")
    print(f"   - HIGH-NEW-HIGH: ~70% (WTP: 1.2-1.6x base)")
    print(f"   - HIGH-RECURRING-HIGH: ~30% (WTP: 1.4-1.8x base)")

    print(f"\nðŸŽ¯ Expected learning:")
    print(f"   - NEW customers: Level 1-2 pricing optimal")
    print(f"   - RECURRING customers: Level 2+ pricing optimal")
    print(f"   - Agent should learn both contexts simultaneously")
    print(f"   - Target performance: 80-85% (vs 79.69% single context)")


if __name__ == '__main__':
    generate_two_context_scenario(
        filepath="two_context_scenario.csv"
    )

    print("\n" + "=" * 60)
    print("TESTING PLAN:")
    print("=" * 60)
    print("âœ… Step 1: Single context (HIGH-NEW-HIGH) â†’ 79.69% âœ…")
    print("ðŸ”„ Step 2: Two contexts (NEW vs RECURRING) â†’ Run this!")
    print("â­ï¸ Step 3: Add commodity variation (HIGH vs LOW)")
    print("â­ï¸ Step 4: Add seasonal variation (HIGH vs MID vs LOW)")
    print("\nðŸ“ˆ Progressive complexity building = Better learning!")