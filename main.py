import numpy as np
import cvxpy as cp
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# Parameters
M = 40  # Number of UAVs (USCs)
N = 20  # Number of antennas per UAV
Wband = 20e6  # Bandwidth = 20 MHz
total_power_mW = 200  # Total available power in mW
no_Monte = 20 

# Load the .mat files containing location and path loss data
channel_data = loadmat('channel1.mat')
uav_user_path_data = loadmat('uavuserpath1.mat')

# Path loss data between UAVs and users
UAV_User_path = uav_user_path_data['UAV_User_path']
Noise_var = 290 * 1.38e-23 * Wband * 10**(9/10)  # Noise variance

# Compute beta_mk (large-scale fading)
beta_mk = UAV_User_path / Noise_var

# Define the user counts to test
user_counts = [10, 15, 20, 25, 30, 35]

# BFOA parameters
num_bacteria = 500
num_iterations = 50
C = 0.05  # Step size for tumbling
d_attract = 0.1
w_attract = 0.2
h_repel = 0.1
w_repel = 0.1


# Initialize result arrays
average_rates_case1, average_rates_case2, average_rates_case3, average_rates_case4, average_rates_case5 = [], [], [], [], []
average_sum_rates_case1, average_sum_rates_case2, average_sum_rates_case3, average_sum_rates_case4, average_sum_rates_case5 = [], [], [], [], []

def swarming_cost(theta, bacteria_positions):
    """
    Calculate the swarming cost based on cell-to-cell signaling.
    """
    swarm_cost = 0
    for i in range(len(bacteria_positions)):
        attract_term = -d_attract * np.exp(-w_attract * np.sum((theta - bacteria_positions[i]) ** 2))
        repel_term = h_repel * np.exp(-w_repel * np.sum((theta - bacteria_positions[i]) ** 2))
        swarm_cost += attract_term + repel_term
    return swarm_cost

def fitness_function(p, K, beta_mk, alpha_star, bacteria_positions):
    """
    Calculate the fitness function for the given power allocation.
    Includes swarming cost.
    """
    rate_sum = 0
    for k in range(K):
        G_k = np.sqrt(beta_mk[:, k]).reshape(-1, 1)  # Simplified channel gains
        interference = np.sum(G_k ** 2 * p) - G_k[k] ** 2 * p[k]
        SINR_k = (G_k[k] ** 2 * p[k]) / (interference + Noise_var)
        rate_k = Wband * np.log2(1 + SINR_k)
        rate_sum += alpha_star[k] * rate_k
    # Add swarming cost
    swarm_cost = swarming_cost(p, bacteria_positions)
    return -rate_sum + swarm_cost  # Negative for minimization

def bfoa_optimization(K, beta_mk, alpha_star):
    total_power_W = total_power_mW / 1000  # Convert total power to Watts
    power_limits = [np.random.uniform(0.01, 0.015, size=K), total_power_W]

    # Initialize bacteria
    bacteria_positions = np.random.uniform(power_limits[0], power_limits[1], size=(num_bacteria, K))
    for i in range(num_bacteria):
        bacteria_positions[i] = bacteria_positions[i] / np.sum(bacteria_positions[i]) * total_power_W

    fitness = np.array([fitness_function(bacteria_positions[i], K, beta_mk, alpha_star, bacteria_positions)
                        for i in range(num_bacteria)])

    for iteration in range(num_iterations):
        for i in range(num_bacteria):
            delta = np.random.uniform(-1, 1, size=K)
            delta /= np.linalg.norm(delta)
            new_position = bacteria_positions[i] + C * delta
            new_position = np.clip(new_position, power_limits[0], power_limits[1])

            # Normalize to ensure total power constraint
            new_position = new_position / np.sum(new_position) * total_power_W

            new_fitness = fitness_function(new_position, K, beta_mk, alpha_star, bacteria_positions)
            if new_fitness < fitness[i]:
                bacteria_positions[i] = new_position
                fitness[i] = new_fitness

        # Reproduction
        J_health = np.argsort(fitness)
        for i in range(num_bacteria // 2):
            bacteria_positions[J_health[-(i + 1)]] = bacteria_positions[J_health[i]]
            fitness[J_health[-(i + 1)]] = fitness[J_health[i]]

        # Elimination and dispersal
        for i in range(num_bacteria):
            if np.random.rand() < 0.1:
                bacteria_positions[i] = np.random.uniform(power_limits[0], power_limits[1], size=K)
                bacteria_positions[i] = bacteria_positions[i] / np.sum(bacteria_positions[i]) * total_power_W
                fitness[i] = fitness_function(bacteria_positions[i], K, beta_mk, alpha_star, bacteria_positions)

    best_index = np.argmin(fitness)
    best_allocation = bacteria_positions[best_index]
    best_fitness = -fitness[best_index]
    return best_allocation, best_fitness


# Monte Carlo simulation
for K in user_counts:
    print(K)
    worst_user_rates_case1 = np.zeros(no_Monte)
    worst_user_rates_case2 = np.zeros(no_Monte)
    worst_user_rates_case3 = np.zeros(no_Monte)
    worst_user_rates_case4 = np.zeros(no_Monte)
    worst_user_rates_case5 = np.zeros(no_Monte)

    sum_user_rates_case1 = np.zeros(no_Monte)
    sum_user_rates_case2 = np.zeros(no_Monte)
    sum_user_rates_case3 = np.zeros(no_Monte)
    sum_user_rates_case4 = np.zeros(no_Monte)
    sum_user_rates_case5 = np.zeros(no_Monte)

    for trial in range(no_Monte):
        beta_hat = np.mean(beta_mk[:K, :K], axis=0)
        alpha_star = beta_hat / np.sum(beta_hat)

        # Case 1: Multi-Criteria Decision for User-Centric (MCUC)
        P1 = cp.Variable(K)
        objective_case1 = cp.Maximize(cp.sum(cp.multiply(alpha_star, Wband * cp.log(1 + cp.multiply(beta_hat, P1)) / np.log(2))))
        constraints_case1 = [cp.sum(P1) <= total_power_mW / 1000, P1 >= 0]
        problem_case1 = cp.Problem(objective_case1, constraints_case1)
        problem_case1.solve()
        worst_user_rates_case1[trial] = min(Wband * np.log2(1 + beta_hat * P1.value) / 1e6)
        sum_user_rates_case1[trial] = sum(Wband * np.log2(1 + beta_hat * P1.value) / 1e6)
        print("case1: ",P1.value)
        # Case 2: Maximin Worst Rate (MMWR)
        P2 = cp.Variable(K)
        SINR_min = cp.Variable()
        constraints_case2 = [P2 >= 0, cp.sum(P2) <= total_power_mW / 1000]
        for k in range(K):
            constraints_case2.append(SINR_min <= beta_hat[k] * P2[k])
        objective_case2 = cp.Maximize(SINR_min)
        problem_case2 = cp.Problem(objective_case2, constraints_case2)
        problem_case2.solve()
        worst_user_rates_case2[trial] = min(Wband * np.log2(1 + beta_hat * P2.value) / 1e6)
        sum_user_rates_case2[trial] = sum(Wband * np.log2(1 + beta_hat * P2.value) / 1e6)
        print("case2: ",P2.value)

        # Case 3: Equal Power Allocation (EPA)
        equal_power_W = total_power_mW / (1000 * K)
        print("case3: ",equal_power_W)
        rates_case3 = Wband * np.log2(1 + beta_hat * equal_power_W) / 1e6
        worst_user_rates_case3[trial] = min(rates_case3)
        sum_user_rates_case3[trial] = sum(rates_case3)

        # Case 4: Random Power Allocation (RPA)
        random_factors = np.random.rand(K)
        random_power_allocation = (random_factors / sum(random_factors)) * (total_power_mW / 1000)
        print("case4: ",random_power_allocation)
        rates_case4 = Wband * np.log2(1 + beta_hat * random_power_allocation) / 1e6
        worst_user_rates_case4[trial] = min(rates_case4)
        sum_user_rates_case4[trial] = sum(rates_case4)

        # Case 5: BFOA Optimization
        best_allocation, _ = bfoa_optimization(K, beta_mk[:K, :K], alpha_star)
        # rate_case5_list = []
        # for k in range(K):
        #     # G_k = np.sqrt(beta_mk[:, k]).reshape(-1, 1)
        #     # interference = np.sum(G_k ** 2 * best_allocation) - G_k[k] ** 2 * best_allocation[k]
        #     # SINR_k = (G_k[k] ** 2 * best_allocation[k]) / (interference + Noise_var)
        #     # rate_k = Wband * np.log2(1 + SINR_k)
        #     rate_case5_list.append(rate_k)
        rates_case5 = Wband * np.log2(1 + beta_hat * best_allocation) / 1e6
        # print(rates_case5)
        worst_user_rates_case5[trial] = min(rates_case5)
        sum_user_rates_case5[trial] = sum(rates_case5)
        print("case5: ",best_allocation)

        # worst_user_rates_case5.append(min(rate_case5_list) / 1e6)
        # sum_user_rates_case5.append(sum(rate_case5_list) / 1e6)

    average_rates_case1.append(np.mean(worst_user_rates_case1))
    average_rates_case2.append(np.mean(worst_user_rates_case2))
    average_rates_case3.append(np.mean(worst_user_rates_case3))
    average_rates_case4.append(np.mean(worst_user_rates_case4))
    average_rates_case5.append(np.mean(worst_user_rates_case5))

    average_sum_rates_case1.append(np.mean(sum_user_rates_case1))
    average_sum_rates_case2.append(np.mean(sum_user_rates_case2))
    average_sum_rates_case3.append(np.mean(sum_user_rates_case3))
    average_sum_rates_case4.append(np.mean(sum_user_rates_case4))
    average_sum_rates_case5.append(np.mean(sum_user_rates_case5))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(user_counts, average_rates_case1, marker='o', label="MCUC")
plt.plot(user_counts, average_rates_case2, marker='s', label="MMWR")
plt.plot(user_counts, average_rates_case3, marker='^', label="EPA")
plt.plot(user_counts, average_rates_case4, marker='x', label="RPA")
plt.plot(user_counts, average_rates_case5, marker='d', label="BFOA")
plt.xlabel("Number of Users (K)")
plt.ylabel("Average Worst-User Rate (Mbps)")
plt.xticks(user_counts)
plt.grid(True)
plt.legend()
plt.savefig("Case_Comparison_WorstRate.png")
plt.show()

print("------------------------------")

plt.figure(figsize=(10, 6))
plt.plot(user_counts, average_sum_rates_case1, marker='o', label="MCUC")
plt.plot(user_counts, average_sum_rates_case2, marker='s', label="MMWR")
plt.plot(user_counts, average_sum_rates_case3, marker='^', label="EPA")
plt.plot(user_counts, average_sum_rates_case4, marker='x', label="RPA")
plt.plot(user_counts, average_sum_rates_case5, marker='d', label="BFOA")
plt.xlabel("Number of Users (K)")
plt.ylabel("Average Sum Rate (Mbps)")
plt.xticks(user_counts)
plt.grid(True)
plt.legend()
plt.savefig("Case_Comparison_SumRate.png")
plt.show()
