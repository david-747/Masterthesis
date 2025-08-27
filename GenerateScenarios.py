import pandas as pd

# Define the filename for clarity
original_scenario_file = 'scenarios/customer_scenario_contextual_wtp.csv'
low_season_scenario_file = 'scenarios/low_season_customer_scenario_contextual_wtp.csv'

try:
    # Load your original scenario file
    df = pd.read_csv(original_scenario_file)

    # --- Create the Low-Season Scenario ---

    # 1. Reduce the number of customer arrivals by 30%
    # We sample 70% of the original arrivals to simulate a quieter season
    df_low_season = df.sample(frac=0.5, random_state=42).copy()
    # Sort by the timestamp 't' to maintain a chronological event sequence
    df_low_season.sort_values(by='t', inplace=True)

    # 2. Set the season context to 'LOW' for all entries
    df_low_season['season'] = 'LOW'

    # 3. Reduce the maximum willingness to pay (max_wtp) by 40%
    df_low_season['max_wtp'] = df_low_season['max_wtp'] * 0.5

    # 4. Adjust the product mix to reflect low-season demand
    df_low_season['REEF'] = (df_low_season['REEF'] * 0.2).round().astype(int)
    df_low_season['TEU'] = (df_low_season['TEU'] * 1.1).round().astype(int)
    df_low_season['FEU'] = (df_low_season['FEU'] * 1.1).round().astype(int)
    df_low_season['HC'] = (df_low_season['HC'] * 1.1).round().astype(int)

    # Save the new, more realistic scenario to a CSV file
    df_low_season.to_csv(low_season_scenario_file, index=False)

    print(f"Successfully created '{low_season_scenario_file}'")
    print(f"Original number of arrivals: {len(df)}")
    print(f"New number of arrivals for low season: {len(df_low_season)}")

except FileNotFoundError:
    print(f"Error: '{original_scenario_file}' not found. Please ensure this file is in the same directory as the script.")