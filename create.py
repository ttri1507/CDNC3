import numpy as np
from scipy.io import savemat, loadmat

# Function to create a circular cell layout (unchanged)
def cell_circle(x0, y0, radius):
    th = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(th) + x0
    y = radius * np.sin(th) + y0
    return x, y, x0, y0

# Function to create a hexagonal cell layout (unchanged)
def cell_hexagon(x0, y0, cote):
    x = cote * np.array([0, 1, 1, 0, -1, -1, 0]) * np.sqrt(3) / 2 + x0
    y = cote * np.array([1, 0.5, -0.5, -1, -0.5, 0.5, 1]) + y0
    return x, y, x0, y0

# Function to generate base station and user locations (unchanged)
def location_bs_user(num_users, rad_bs):
    x0, y0 = 0, 0
    x, y, XBS, YBS = cell_circle(x0, y0, rad_bs)

    Dtempt_Users = 200 * np.sqrt(np.random.rand(num_users)) + 250
    Angtemp_Users = np.random.rand(num_users) * 2 * np.pi
    X_Users = Dtempt_Users * np.cos(Angtemp_Users)
    Y_Users = Dtempt_Users * np.sin(Angtemp_Users)

    return XBS, YBS, X_Users, Y_Users

# New function to generate the UAV locations
def generate_uav_locations(num_uavs, height, radius):
    Dtempt_UAVs = radius * np.sqrt(np.random.rand(num_uavs))
    Angtemp_UAVs = np.random.rand(num_uavs) * 2 * np.pi
    X_UAVs = Dtempt_UAVs * np.cos(Angtemp_UAVs)
    Y_UAVs = Dtempt_UAVs * np.sin(Angtemp_UAVs)
    Z_UAVs = np.ones(num_uavs) * height  # All UAVs have the same height

    return X_UAVs, Y_UAVs, Z_UAVs

# Modified path loss function to handle multiple UAVs
def func_path_multiple_uavs(num_uavs, num_users, X_UAVs, Y_UAVs, Z_UAVs, X_Users, Y_Users, Wband, eta, L, sigma_n):
    pl_exponent = 3  # path-loss exponent
    beta_0 = 10 ** (-30 / 10)  # Channel power gain at the reference distance
    a = 11.95
    b = 0.136
    gamma = 10 ** (20 / 10)  # The excessive attenuation factor

    # Create a matrix to store the path loss between each UAV and each user
    UAV_User_path = np.zeros((num_uavs, num_users))

    # Loop over each UAV and each user to calculate the path loss
    for uav in range(num_uavs):
        for user in range(num_users):
            # Calculate the angle phi (elevation angle)
            distance = np.sqrt((X_Users[user] - X_UAVs[uav]) ** 2 +
                               (Y_Users[user] - Y_UAVs[uav]) ** 2 +
                               Z_UAVs[uav] ** 2)
            phi = np.degrees(np.arcsin(Z_UAVs[uav] / distance))

            # Calculate probabilities of LOS and NLOS
            Pr_los = 1 / (1 + a * np.exp(-b * (phi - a)))
            Pr_nlos = 1 - Pr_los

            # Calculate path loss for LOS and NLOS
            UAV_User_path[uav, user] = Pr_los * (distance) ** (-4) + Pr_nlos * gamma * (distance) ** (-4)

    return UAV_User_path

# Main function to run the simulation with multiple UAVs
def main():
    # Simulation parameters
    Wband = 20e6  # MHz
    RadBS = 1000  # radius coverage in meters
    height = 200  # height of UAVs in meters
    NumUsers_ref = 35  # number of users
    NumUAVs = 40  # number of UAVs
    eta = 0.5  # energy harvesting efficiency (not used)
    L = 1  # number of antennas at each user
    Noise_var = 10 ** (-130 / 10) * Wband * 10 ** 6  # noise variance
    sigma_n = Noise_var  # set noise variance

    # Generate the location of the base station and users
    XBS, YBS, X_Users, Y_Users = location_bs_user(NumUsers_ref, RadBS)

    # Generate the locations of multiple UAVs
    X_UAVs, Y_UAVs, Z_UAVs = generate_uav_locations(NumUAVs, height, RadBS)

    # Save the location data into a .mat file
    savemat('channel1.mat', {
        'XBS': XBS,
        'YBS': YBS,
        'X_Users': X_Users,
        'Y_Users': Y_Users,
        'X_UAVs': X_UAVs,
        'Y_UAVs': Y_UAVs,
        'Z_UAVs': Z_UAVs
    })

    # Calculate the path loss between each UAV and each user
    UAV_User_path = func_path_multiple_uavs(NumUAVs, NumUsers_ref, X_UAVs, Y_UAVs, Z_UAVs, X_Users, Y_Users, Wband, eta, L, sigma_n)

    # Save the path loss data into a .mat file
    savemat('uavuserpath1.mat', {
        'UAV_User_path': UAV_User_path
    })

    print("Data saved to channel.mat and uavuserpath.mat")

if __name__ == "__main__":
    main()
