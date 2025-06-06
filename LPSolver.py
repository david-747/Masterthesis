import pulp
import numpy as np

# Import classes for type hinting
from MiscShipping import Product, PriceVector
from Context import Context


def solve_real_lp(
        #agents current guess for the mean demand of each product at every price in every context
        #comes from sampling from the internal model DelayedTSAgent
        sampled_theta_t: dict,
        #represents average per-period budget for each resource
        #calculated as total initial inventory of a resource divided by total number of time periods
        resource_constraints_c_j: np.ndarray,
        #all possible situations the model can encounter
        all_contexts: list[Context],
        #all products that can be sold
        all_products: list[Product],
        #unique ids for each price vector
        all_price_indices: list[int],
        #mapping id to actual price vector
        all_price_vectors_map: dict[int, PriceVector],
        #matrix specifying how many units of resource j are consumed by one unit of product i
        resource_consumption_matrix_A_ij: np.ndarray,
        context_probabilities: dict | None,
        #helper dict to quickly find correct row in the ressource consumption matrix given product id
        product_to_idx_map: dict
) -> dict:
    """
    A real LP solver based on Ferreira et al. (2018) Algorithm 1,
    solved independently for each context.

    This function is a central component of the pricing engine and is designed
    to be independent of any simulation environment.
    """
    lp_solution = {}
    num_resources = len(resource_constraints_c_j)

    # Solve an independent LP for each context
    for context in all_contexts:
        # 1. Create the LP problem
        #prob = pulp.LpProblem(f"RevenueMax_for_Context_{context}", pulp.LpMaximize)

        prob = pulp.LpProblem(f"RevenueMax_for_Context_{context.to_safe_str()}", pulp.LpMaximize)

        # 2. Define decision variables (x_k in the paper)
        # The probability of choosing each price vector k
        prob_variables = pulp.LpVariable.dicts(
            "Prob_PV", all_price_indices, lowBound=0, upBound=1
        )

        # 3. Define the Objective Function
        # Maximize: sum over k ( (sum over i (p_ik * d_ik)) * x_k )
        revenue_per_pv = {}
        for pv_id in all_price_indices:
            total_revenue_for_pv = 0
            for product in all_products:
                # Get price p_ik
                price_p_ik = all_price_vectors_map[pv_id].get_price_object(product).amount
                # Get sampled demand d_ik for the current context
                demand_d_ik = sampled_theta_t[context][product][pv_id]
                total_revenue_for_pv += price_p_ik * demand_d_ik
            revenue_per_pv[pv_id] = total_revenue_for_pv

        prob += pulp.lpSum(
            [revenue_per_pv[pv_id] * prob_variables[pv_id] for pv_id in all_price_indices]
        ), "Total_Expected_Revenue"

        # 4. Define Constraints
        # Resource constraints: sum over k ( (sum over i (a_ij * d_ik)) * x_k ) <= c_j
        for j in range(num_resources):
            consumption_per_pv = {}
            for pv_id in all_price_indices:
                total_consumption_for_pv = 0
                for product in all_products:
                    # Get product index for the consumption matrix
                    i = product_to_idx_map[product.product_id]
                    consumption_a_ij = resource_consumption_matrix_A_ij[i, j]
                    demand_d_ik = sampled_theta_t[context][product][pv_id]
                    total_consumption_for_pv += consumption_a_ij * demand_d_ik
                consumption_per_pv[pv_id] = total_consumption_for_pv

            prob += pulp.lpSum(
                [consumption_per_pv[pv_id] * prob_variables[pv_id] for pv_id in all_price_indices]
            ) <= resource_constraints_c_j[j], f"Resource_Constraint_{j}"

        # Probability constraint: sum of x_k <= 1
        prob += pulp.lpSum([prob_variables[pv_id] for pv_id in all_price_indices]) <= 1, "Sum_of_Probs_Constraint"

        # 5. Solve the LP
        # Use a silent solver so it doesn't print logs during the simulation
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # 6. Store the results in the required format
        probabilities_for_context = {}
        for pv_id in all_price_indices:
            probabilities_for_context[pv_id] = prob_variables[pv_id].varValue
        lp_solution[context] = probabilities_for_context

    return lp_solution