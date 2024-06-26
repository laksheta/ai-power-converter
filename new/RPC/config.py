import pandas as pd
import numpy as np
from calc import read_and_sample_csv
import itertools


V_in = 48
V_out = 12

# Parameters
R_ON_H = 5.6e-3  # 5.6 milliohms
R_ON_L = 5.6e-3  # 5.6 milliohms

V_D = 0.7  # Diode forward voltage in Volts

C_OSS_H = 540e-12  # 540 pF in Farads

Q_g_H = Q_g_L = 170e-9  # 170 nC in Coulombs
V_GS = 10    

d = 0.1  # diameter wire (mm)
rho = 1.68e-5  # resistivitas (Ohm.mm)

tan_delta = 0.14

t_d_range = (175.2e-9, 262.8e-9)  # Range for dead time
fsw_range = (20e3, 200e3)  # Range for fsw
d_cycle_range = (0.2, 0.35)
num_values = 2  # Number of values for each parameter

# Load lookup tables
inductor_lookup_table = r'E:\ai-power-converter-1\new\dataset\lookup_inductor_new.csv'
capacitor_lookup_table = r'E:\ai-power-converter-1\new\dataset\lookup_capacitor_new.csv'
data_inductor = pd.read_csv(inductor_lookup_table)



# Get sampled values for L and C
L_values = read_and_sample_csv(inductor_lookup_table, 'L(uH)', num_values)
C_values = read_and_sample_csv(capacitor_lookup_table, 'Cap(uF)', num_values)
fsw_values = np.linspace(fsw_range[0], fsw_range[1], num=num_values)
t_d_values = np.linspace(t_d_range[0], t_d_range[1], num=num_values)
d_cycle_values = np.linspace(d_cycle_range[0], d_cycle_range[1], num=num_values)

# Print the chosen values
print("Chosen L values:", L_values)
print("Chosen C values:", C_values)
print("Chosen dead time values:", t_d_values)
print("Chosen fsw values:", fsw_values)
print("Chosen d_cycle values:", d_cycle_values)


# Generate all combinations of L, C, and fsw
val_tests = list(itertools.product(L_values, C_values, fsw_values, t_d_values, d_cycle_values))

# Set up simulation parameters
# val_tests = [
#     [30, 15, 250e3, 20e-9, 0.1],
#     [20, 25, 250e3, 20e-9, 0.2]
# ]

# Print the number of combinations and the combinations themselves
num_combinations = len(val_tests)
print(f"Number of combinations: {num_combinations}")
print("Parameter combinations:")
for i, combination in enumerate(val_tests):
    print(f"{i+1}: {combination}")


# Define the start and end times for the output
start_time = 0.004#0.0
end_time = 0.005#1e-2
step_size = 1e-8
