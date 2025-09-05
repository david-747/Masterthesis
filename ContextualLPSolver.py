import pulp
import numpy as np
from Context import Context
from MiscShipping import *

def solve_contextual_lp(
        sampled_theta_t: dict,
        resource_constraints_c_j: np.ndarray,
        all_contexts: list[Context],
        all_products: list[Product],
        all_price_indices: list[int],
        all_price_vectors_map: dict[int, PriceVector],
        resource_consumption_matrix_A_ij: np.ndarray,
        demand_scaling_factor: float,
        context_probabilities: dict[Context, float],  # This should NOT be None
        product_to_idx_map: dict
) -> dict:
    """
    LP solver for contextual setting based on Algorithm 4 from Ferreira et al.
    Solves a single global LP considering all contexts.
    """

    # Create the LP problem
    prob = pulp.LpProblem("Contextual_RevenueMax", pulp.LpMaximize)

    # Decision variables: x_{context,price_vector}
    # For each context and price vector combination
    x_vars = {}
    for context in all_contexts:
        x_vars[context] = {}
        for pv_id in all_price_indices:
            x_vars[context][pv_id] = pulp.LpVariable(
                f"x_{context.to_safe_str()}_{pv_id}",
                lowBound=0,
                upBound=1
            )
    '''
    prob_floor_tau = 0.4 #0.01  # e.g., 1% — tune 0.5–2%
    
    for context in all_contexts:
        for pv_id in all_price_indices:
            d_vals = []
            for prod in all_products:
                # Keep this consistent with your objective:
                # sampled_theta_t[context][product][pv_id]
                prob_d_ik = sampled_theta_t[context][prod][pv_id]
                d_vals.append(prob_d_ik)
            score = max(d_vals)
            if score < prob_floor_tau:
                x_vars[context][pv_id].upBound = 0.0
'''
    # Objective: Maximize expected revenue across all contexts
    # E_ξ[sum_k sum_i p_ik * d_ik(ξ|θ) * x_ξ,k]
    objective = 0
    for context in all_contexts:
        context_prob = context_probabilities.get(context, 1.0 / len(all_contexts))
        for pv_id in all_price_indices:
            revenue_for_context_price = 0
            for product in all_products:
                price_p_ik = all_price_vectors_map[pv_id].get_price_object(product).amount
                prob_d_ik = sampled_theta_t[context][product][pv_id]
                expected_quantity = prob_d_ik * demand_scaling_factor
                revenue_for_context_price += price_p_ik * expected_quantity

            objective += context_prob * revenue_for_context_price * x_vars[context][pv_id]

    prob += objective, "Expected_Revenue"

    # Resource constraints: E_ξ[sum_k sum_i a_ij * d_ik(ξ|θ) * x_ξ,k] <= c_j
    for j in range(len(resource_constraints_c_j)):
        resource_constraint = 0
        for context in all_contexts:
            context_prob = context_probabilities.get(context, 1.0 / len(all_contexts))
            for pv_id in all_price_indices:
                consumption_for_context_price = 0
                for product in all_products:
                    i = product_to_idx_map[product.product_id]
                    consumption_a_ij = resource_consumption_matrix_A_ij[i, j]
                    prob_d_ik = sampled_theta_t[context][product][pv_id]
                    expected_quantity = prob_d_ik * demand_scaling_factor
                    consumption_for_context_price += consumption_a_ij * expected_quantity

                resource_constraint += context_prob * consumption_for_context_price * x_vars[context][pv_id]

        prob += resource_constraint <= resource_constraints_c_j[j], f"Resource_Constraint_{j}"

    # Probability constraints: For each context, sum of x_ξ,k <= 1
    for context in all_contexts:
        prob += pulp.lpSum([x_vars[context][pv_id] for pv_id in all_price_indices]) <= 1, \
            f"Prob_Constraint_{context.to_safe_str()}"

    # Solve the LP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract solution
    lp_solution = {}
    for context in all_contexts:
        lp_solution[context] = {}
        for pv_id in all_price_indices:
            lp_solution[context][pv_id] = x_vars[context][pv_id].varValue or 0.0

    return lp_solution