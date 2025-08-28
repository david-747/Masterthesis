import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

# --- Configuration ---
RUN_FOLDER = "simulation_outputs_contextual/run_2025-08-28_11-28-42" # <-- IMPORTANT: CHANGE THIS
METRICS_FILE = [f for f in os.listdir(RUN_FOLDER) if f.startswith('metrics_log')][0]
BELIEFS_FILE = "agent_beliefs.json"

metrics_path = os.path.join(RUN_FOLDER, METRICS_FILE)
beliefs_path = os.path.join(RUN_FOLDER, BELIEFS_FILE)

# --- Analysis ---
df_metrics = pd.read_csv(metrics_path)
with open(beliefs_path, 'r') as f:
    beliefs = json.load(f)

print(f"Analyzing results from {RUN_FOLDER}...")

# --------------------------------------------------------------------------
# --- 1. Visualize Agent's Learned Beliefs about PRICE VECTORS           ---
# --- NEW SIMPLIFIED VERSION                                             ---
# --------------------------------------------------------------------------

# We only need to look at one product's data since they are all the same.
# Let's just pick 'TEU' to represent the belief for the entire price vector.
prob_data = []
for context, arms in beliefs.items():
    for arm_key, belief in arms.items():
        product_id, price_idx = arm_key.split('-')
        if product_id == 'TEU': # Using TEU as the representative for the price vector
            prob_data.append({
                'context': context,
                'price_vector_id': int(price_idx),
                'prob_acceptance': belief['prob_success']
            })

df_probs = pd.DataFrame(prob_data)
pivot_probs = df_probs.pivot(index='context', columns='price_vector_id', values='prob_acceptance')

# Create the single, consolidated heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_probs, annot=True, cmap="viridis", fmt=".2%", linewidths=.5)
plt.title("Agent's Learned Acceptance Probability for each Price Vector", fontsize=16, pad=20)
plt.xlabel("Price Vector (0=Aggressive, 1=Standard, 2=Premium)", fontsize=12)
plt.ylabel("Customer Context", fontsize=12)
plt.yticks(rotation=0)
plt.tight_layout()

heatmap_path = os.path.join(RUN_FOLDER, "analysis_heatmap_price_vector_beliefs.png")
plt.savefig(heatmap_path)
print(f"-> Consolidated Price Vector Beliefs heatmap saved to {heatmap_path}")
#plt.show()


# --------------------------------------------------------------------------
# --- 2. Analyze Price Selections per Context (No Changes)               ---
# --------------------------------------------------------------------------
df_agent = df_metrics[df_metrics['phase'] == 'agent'].copy()

plt.figure(figsize=(12, 7))
sns.countplot(
    data=df_agent,
    x='context_commodity_value',
    hue='chosen_pv_id',
    palette='pastel'
)
plt.title("Price Vectors Chosen for High vs. Low Value Commodities", fontsize=16)
plt.xlabel("Commodity Value", fontsize=12)
plt.ylabel("Number of Times Offered", fontsize=12)
plt.legend(title='Price Vector ID (0: Aggressive, 1: Standard, 2: Premium)')
plt.tight_layout()
countplot_path = os.path.join(RUN_FOLDER, "analysis_countplot_choices.png")
plt.savefig(countplot_path)
print(f"\n-> Choice count plot saved to {countplot_path}")
#plt.show()