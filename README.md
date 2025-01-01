# **CDNC3**  

This repository implements a simulation for power allocation optimization using the Bacterial Foraging Optimization Algorithm (BFOA) and other comparative methods such as MCUC, MMWR, EPA, and RPA.

## **Structure**
- `main.py`: Main script to run the simulation and generate performance results (e.g., plots of worst-user rate and sum rate).
- `create.py`: Script for generating channel data and saving it into `.mat` files.
- `channel1.mat`: Pre-generated channel matrix used for calculations (large-scale fading).
- `uavuserpath1.mat`: Pre-generated UAV-user path data.
- `Case_Comparison_SumRate.png`: Plot of average sum rate for different methods.
- `Case_Comparison_WorstRate.png`: Plot of average worst-user rate for different methods.

## **Requirements (System Running)**
To run the code, ensure you have the following dependencies installed:
- Python 3.10.12
- Required Python libraries:
  - `numpy==1.26.4`
  - `scipy==1.13.1`
  - `cvxpy==1.6.0`
  - `matplotlib==3.8.0`

## Install dependencies using:
`pip install numpy==1.26.4 scipy==1.13.1 cvxpy==1.6.0 matplotlib==3.8.0`

## Usage
### Step 1: Generate Channel Data
Run the create.py script to generate the channel matrices (channel1.mat and uavuserpath1.mat):
`python3 create.py`
This script will create .mat files containing the channel and UAV-user path data, which are necessary for running the simulation.

### Step 2: Run the Simulation
Use the main.py script to run the power allocation simulation and generate results:
`python3 main.py`
This will:
  - Calculate and compare the performance of different methods (MCUC, MMWR, EPA, RPA, and BFOA).
  - Generate performance plots:
  - Case_Comparison_SumRate.png: Average sum rate for different methods.
  - Case_Comparison_WorstRate.png: Average worst-user rate for different methods.

### Step 3: View Results
The generated plots are saved in the repository folder:
  - Worst-User Rate: Compare fairness across methods.
  - Sum Rate: Compare the overall channel capacity across methods.

##### Customization
You can modify the parameters such as the number of users, UAV antennas, and total power in the main.py script.
