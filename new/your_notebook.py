#!/usr/bin/env python
# coding: utf-8

# # Spesifikasi

# - V<sub>in</sub> = 48V
# - V<sub>out</sub> = 12V
# - P<sub>o</sub> = 100W
# - &#916;V<sub>o</sub> &#8804; 1%
# - &#916;I<sub>L</sub> &#8804; 10%
# - f<sub>sw</sub> = 20e3
# - Mosfet: IRFB4310PbF, Infineon

# # Plecs

# In[59]:


import xmlrpc.client as xml
import time
import numpy as np
import itertools
import csv
import pandas as pd


# In[60]:


model = 'buck_new'
file_type = '.plecs'
V_in = 48
V_out = 12
f_sw = 20e3


# In[61]:


plecs = xml.Server("http://localhost:1080/RPC2").plecs


# In[62]:


plecs.load(r"E:\ai-power-converter-1\new\buck_new.plecs")


# In[63]:


plecs.get(model+'/FETD1')


# In[64]:


plecs.get(model+'/FETD2')


# In[65]:


plecs.get(model+'/L1')


# In[66]:


plecs.get(model+'/DCR')


# In[67]:


plecs.get(model+'/ESR')


# In[68]:


plecs.get(model+'/C')


# In[69]:


plecs.get(model+'/Load')


# In[70]:


plecs.get(model+'/Symmetrical PWM')


# In[71]:


plecs.get(model+'/Duty Cycle')


# In[72]:


plecs.get(model+'/Deadtime')


# In[73]:


def read_and_sample_csv(file_path, column_name, num_values):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Extract unique values from the specified column
    unique_values = data[column_name].unique()

    # Sort the unique values to ensure proper distribution
    unique_values.sort()

    # Sample num_values evenly spaced values
    if len(unique_values) > num_values:
        indices = np.linspace(0, len(unique_values) - 1, num=num_values, dtype=int)
        sampled_values = unique_values[indices]
    else:
        sampled_values = unique_values  # If fewer values than num_values, use all

    return sampled_values


# # Initialization

# In[95]:


V_in = 48
V_out = 12

#Parameters
R_ON_H = 5.6e-3  # 5.6 milliohms
R_ON_L = 5.6e-3  # 5.6 milliohms

V_D = 0.7  # Diode forward voltage in Volts

C_OSS_H = 540e-12  # 540 pF in Farads

Q_g_H = Q_g_L = 170e-9  # 170 nC in Coulombs
V_GS = 10    


d = 0.1  # diameter wire (mm)
rho = 1.68e-5  # resistivitas (Ohm.mm)

tan_delta = 0.14


t_d_range = (175.2e-9, 262.8e-9) # Range for dead time
fsw_range = (20e3, 200e3)    # Range for fsw
d_cycle_range = (0.2, 0.35)
num_values =  4         # Number of values for each parameter


# Load lookup tables
inductor_lookup_table = r'E:\ai-power-converter-1\new\dataset\lookup_inductor_new.csv'
capacitor_lookup_table = r'E:\ai-power-converter-1\new\dataset\lookup_capacitor_new.csv'
data_inductor = pd.read_csv(inductor_lookup_table)

# Get sampled values for L and C
L_values = read_and_sample_csv(inductor_lookup_table, 'L(uH)', num_values)
C_values = read_and_sample_csv(capacitor_lookup_table, 'Cap(uF)', num_values)
fsw_values = np.linspace(fsw_range[0], fsw_range[1], num=num_values, dtype=int)
t_d_values = np.linspace(t_d_range[0], t_d_range[1], num=num_values)
d_cycle_values = np.linspace(d_cycle_range[0], d_cycle_range[1], num=num_values)


# Round the values to the desired number of decimal places
t_d_values = np.around(t_d_values, decimals=10)

# Initialize arrays to store results
average_currents = []
delta_currents = []
delta_currents_percentage = []

# Open a CSV file for writing
csv_file_path = 'simulation_results_4.csv'

# Define the header for the CSV file
csv_header = ['No', 'L', 'C', 'fsw', 't_d', 'd_cycle', 'average_current', 'delta_current', 'delta_current_percentage', 'average_voltage', 'delta_voltage', 'delta_voltage_percentage', 'DCR', 'ESR', 'P_ON_H', 'P_ON_L', 'P_COSS', 'P_L_DCR', 'P_D', 'P_G',  'P_CAP_ESR', 'P_total']

# Initialize a list to store the data for each simulation
csv_data = []

# Print the chosen values
print("Chosen L values:", L_values)
print("Chosen C values:", C_values)
print("Chosen dead time values:", t_d_values)
print("Chosen fsw values:", fsw_values)

# Generate all combinations of L, C, and fsw
combinations =list(itertools.product(L_values, C_values, fsw_values, t_d_values, d_cycle_values))
print(combinations)
# Display the chosen values and simulate
print("\nChosen values and simulation results:")


# # Functions

# ## Write to CSV

# In[75]:


def write_to_csv(file_path, header, data):
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        csv_writer.writerows(data)


# ## Read and Sample CSV (L and C)

# In[76]:


def read_and_sample_csv(file_path, column_name, num_values):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Extract unique values from the specified column
    unique_values = data[column_name].unique()

    # Sort the unique values to ensure proper distribution
    unique_values.sort()

    # Sample num_values evenly spaced values
    if len(unique_values) > num_values:
        indices = np.linspace(0, len(unique_values) - 1, num=num_values, dtype=int)
        sampled_values = unique_values[indices]
    else:
        sampled_values = unique_values  # If fewer values than num_values, use all

    return sampled_values


# ## DCR

# $$
# DCR = N \times (OD - ID + 2H) \times r \quad [\Omega]
# $$
# 
# $$
# r = \frac{\rho \cdot l}{A} \quad [\Omega]
# $$
# 
# Panjang wire:
# $$
# l = N \times \left( \frac{OD + ID}{2} \times \pi \right) 
# $$
# 

# ## Ripples

# ### Current
# - Average
# - &Delta;I<sub>L</sub>%

# In[77]:


def calculate_current(times, current, start_time, end_time):
    # Convert 'times' and 'current' to NumPy arrays
    times = np.array(times)
    current = np.array(current)

    # Find the indices corresponding to the time range
    start_index = np.argmax(times >= start_time)
    end_index = np.argmax(times >= end_time)
    
    # Ensure the end_index is correctly set to include the end_time value
    if end_index == 0 and times[end_index] < end_time:
        end_index = len(times)

    # Extract the current values in the specified time range
    current_range = current[start_index:end_index]

    # Calculate the average current value
    average_current = np.mean(current_range)

    # Calculate the difference between the highest and lowest current values
    delta_current = np.max(current_range) - np.min(current_range)

    # Calculate the percentage difference
    delta_current_percentage = (delta_current / average_current) * 100

    return average_current, delta_current, delta_current_percentage


# ### &Delta;V<sub>O</sub>%

# In[78]:


def calculate_voltage(times, voltage, start_time, end_time):
    
    times = np.array(times)
    voltage = np.array(voltage)

    # Find the indices corresponding to the time range
    start_index = np.argmax(times >= start_time)
    end_index = np.argmax(times > end_time)

    # Ensure the end_index is correctly set to include the end_time value
    if end_index == 0 and times[end_index] < end_time:
        end_index = len(times)

    # Extract the voltage values in the specified time range
    voltage_range = voltage[start_index:end_index]

    # Calculate the average voltage value
    average_voltage = np.mean(voltage_range)

    # Calculate the difference between the highest and lowest voltage values
    delta_voltage = np.max(voltage_range) - np.min(voltage_range)
    # print('Voltage:')
    # print("\nmax:", np.max(voltage_range))
    # print("\nmin:", np.min(voltage_range))
    # print("\ndelta:",delta_voltage)
    # print("\naverage:",average_voltage)

    delta_voltage_percentage = (delta_voltage/average_voltage) * 100

    return average_voltage, delta_voltage, delta_voltage_percentage


# ## 1. Conduction Losses    
# $$
# P_{ON-H} = I_{out}^2 \times R_{ON-H} \times \frac{V_{out}}{V_{IN}} \quad \text{[\(W\)]}
# $$
# 
# $$
# P_{ON-L} = I_{OUT}^2 \times R_{ON-L} \times \left( 1 - \frac{V_{OUT}}{V_{IN}} \right) \quad \text{[\(W\)]}
# $$
# 
# $$
# R_{ON-H}=R_{ON-L}=5.6m\Omega
# $$

# In[79]:


def calculate_conduction_loss(I_out, V_out, V_IN, R_ON_H, R_ON_L):
    # Calculate P_ON-H
    P_ON_H = I_out**2 * R_ON_H * (V_out / V_IN)
    
    # Calculate P_ON-L
    P_ON_L = I_out**2 * R_ON_L * (1 - (V_out / V_IN))
    
    return P_ON_H, P_ON_L


# ## 2. Switching Loss (bisa diskip)
# 
# $$
# P_{SW-H} = \frac{1}{2} \times V_{in} \times I_{out} \times (t_{r-H} + t_{f-H}) \times f_{sw} \quad \text{[W]}
# $$
# $$
# P_{SW-L} = \frac{1}{2} \times V_{D} \times I_{OUT} \times (t_{r-L} + t_{f-L}) \times f_{SW} \quad \text{[W]}
# $$
# $$
# t_{r-H} = t_{r-L} = 110\text{ns} 
# $$
# $$
# t_{f-H}  = t_{f-L} = 78\text{ns} 
# $$
# $$
# V_D  = 0.7\text{V}
# $$
# 

# In[80]:


def calculate_switching_loss(V_in, I_out, f_sw, t_r_H, t_r_L, t_f_H, t_f_L, V_D):
    # Calculate P_SW-H
    P_SW_H = 0.5 * V_in * I_out * (t_r_H + t_f_H) * f_sw

    # Calculate P_SW-L
    P_SW_L = 0.5 * V_D * I_out * (t_r_L + t_f_L) * f_sw

    return P_SW_H, P_SW_L


# ## 3. Output Capacitance Loss
# $$
# P_{\text{COSS}} = \frac{1}{2} \times C_{\text{OSS-H}} \times V_{\text{IN}}^2 \times f_{\text{SW}} \quad [W]
# $$
# 
# $$
# C_{OSS-H}=540pF

# In[81]:


def calculate_output_capacitance_loss(V_IN, f_sw, C_OSS_H):
    # Calculate P_COSS
    P_COSS = 0.5 * C_OSS_H * V_IN**2 * f_sw

    return P_COSS


# ## 4. Dead Time Loss
# $$
# P_{D} = V_{D} \times I_{\text{OUT}} \times t_d \times f_{\text{SW}} \quad [W]
# $$
# $$
# V_D = 0.7\text{V}
# $$

# In[82]:


def calculate_dead_time_loss(V_D, I_out, f_sw, t_d):

    # Calculate P_D
    P_D = V_D * I_out * t_d * f_sw

    return P_D


# ## 5. Gate Charge Loss
# 
# $$
# P_{G} = (Q_{g-H} + Q_{g-L}) \times V_{gs} \times f_{\text{SW}} \quad [W]
# $$
# $$
# Q_{g-H} = Q_{g-L} = 170nC \\
# $$
# $$
# V_{GS} = 10\text{V}
# $$

# In[83]:


def calculate_gate_charge_loss(f_sw, Q_g_H, Q_g_L, V_GS):

    # Calculate P_G
    P_G = (Q_g_H + Q_g_L) * V_GS * f_sw

    return P_G


# ## 6. Inductor Conduction Loss
# $$
# P_{L(DCR)} = I_{\text{OUT}}^2 \times \text{DCR} \quad [W]
# $$

# In[84]:


def calculate_dcr(input_L, data, d, rho):
    # Menghitung luas penampang kawat
    A = np.pi * (d / 2) ** 2  # luas wire (mm^2)

    # Menentukan baris yang paling mendekati input_L
    data['L_difference'] = abs(data['L(uH)'] - input_L)
    nearest_L_index = data['L_difference'].idxmin()
    selected_row = data.loc[nearest_L_index]

    N = selected_row['N']
    OD = selected_row['OD(mm)']
    ID = selected_row['ID(mm)']
    H = selected_row['Ht(mm)']

    # Menghitung panjang kawat
    l = N * (OD + ID) * np.pi * 0.5

    # Menghitung resistansi kawat
    r = rho * l / A

    # Menghitung DCR
    DCR = N * (OD - ID + 2 * H) * r

    # Menampilkan hasil
    return DCR

def calculate_inductor_conduction_loss(I_out, DCR):

    # Calculate P_L(DCR)
    P_L_DCR = I_out**2 * DCR

    return P_L_DCR


# ## 7. Capacitor Loss
# $$
# P_{\text{CAP(ESR)}} = I_{\text{CAP(RMS)}}^2 \times \text{ESR} \quad [W]
# $$
# 
# $$
# \text{ESR} = \frac{\tan \delta}{2 \pi f_s C}
# $$
# 
# $$
# tan\delta = 0.14
# $$

# In[85]:


def calculate_I_CAP_RMS(delta_IL):
    # Calculate I_CAP_RMS
    I_CAP_RMS = delta_IL / (2 * np.sqrt(3))
    return I_CAP_RMS

def calculate_esr(tan_delta, f_s, C):
    # Calculate ESR
    ESR = tan_delta / (2 * np.pi * f_s * C)
    return ESR

def calculate_capacitor_loss(delta_IL, ESR):
    # Calculate I_CAP_RMS
    I_CAP_RMS = calculate_I_CAP_RMS(delta_IL)

    # Calculate P_CAP(ESR)
    P_CAP_ESR = I_CAP_RMS**2 * ESR
    
    return P_CAP_ESR


# ## Total Losses
# 
# $$
# P = P_{ON-H} + P_{ON-L} + P_{SW-H} + P_{SW-L} + P_{COSS} + P_{D} + P_{G} + P_{L(DCR)} + P_{CAP(ESR)} \quad [W]
# $$

# In[86]:


def calculate_total_losses(P_ON_H, P_ON_L, P_COSS, P_D, P_G, P_L_DCR, P_CAP_ESR):
    P_total = P_ON_H + P_ON_L  + P_COSS + P_D + P_G + P_L_DCR + P_CAP_ESR
    return P_total


# # Single

# In[87]:


val_test =[20, 25, 250e3, 20e-9, 0.2]

L = val_test[0]*1e-6
C = val_test[1]*1e-6
fsw = val_test[2]
t_d = val_test[3]
d_cycle = val_test[4]

DCR = calculate_dcr(L, data_inductor, d, rho)*1e-3
ESR = calculate_esr(tan_delta, fsw, C)

# Set Plecs parameters and simulate
plecs.set(model+'/FETD1', 'Ron', str(R_ON_H))
plecs.set(model+'/FETD2', 'Ron', str(R_ON_L))
plecs.set(model+'/L1', 'L', str(L))
plecs.set(model+'/DCR', 'R', str(DCR))
plecs.set(model+'/ESR', 'R', str(ESR))
plecs.set(model+'/C', 'C', str(C))
plecs.set(model+'/Symmetrical PWM', 'fc', str(fsw))
plecs.set(model+'/Deadtime', 'td', str(t_d))
plecs.set(model+'/Duty Cycle', 'Value', str(d_cycle))
times = plecs.simulate(model)['Time']
current = plecs.simulate(model)['Values'][0]
voltage = plecs.simulate(model)['Values'][1]

# Calculate average and delta current using the function
average_current, delta_current, delta_current_percentage = calculate_current(times, current, start_time=0.004, end_time=0.005)
average_voltage, delta_voltage, delta_voltage_percentage = calculate_voltage(times, voltage, start_time=0.004, end_time=0.005)
P_ON_H, P_ON_L = calculate_conduction_loss(average_current, average_voltage, V_in, R_ON_H, R_ON_L)
P_COSS = calculate_output_capacitance_loss(V_in, fsw, C_OSS_H)
P_D = calculate_dead_time_loss(V_D, average_current, fsw, t_d)
P_G = calculate_gate_charge_loss(fsw, Q_g_H, Q_g_L, V_GS)
P_L_DCR = calculate_inductor_conduction_loss(average_current, DCR)
P_CAP_ESR = calculate_capacitor_loss(delta_current, ESR)
P_total = calculate_total_losses(P_ON_H, P_ON_L, P_COSS, P_D, P_G, P_L_DCR, P_CAP_ESR)

print(f"Average Current between 0.004 and 0.005 seconds: {average_current} A")
print(f"Difference between highest and lowest current values between 0.004 and 0.005 seconds: {delta_current} A")
print(f"Difference between highest and lowest current values between 0.004 and 0.005 seconds: {delta_current_percentage} %")
print(f"Difference between highest and lowest voltage values between 0.004 and 0.005 seconds: {delta_voltage} V")
print(f"Difference between highest and lowest voltage values between 0.004 and 0.005 seconds: {delta_voltage_percentage} %")
print(f"Conduction Loss: {P_ON_H} and {P_ON_L} W")
print(f"Output Capacitance Loss: {P_COSS} W")
print(f"Dead Time Loss: {P_D} W")
print(f"Gate Charge Loss: {P_G} W")
print(f"Inductor Conduction Loss: {P_L_DCR} W")
print(f"Capacitor Loss: {P_CAP_ESR} W")
print(f"Total Losses: {P_total} W")


# ## Cek waktu
# 

# In[88]:


import time

val_tests = [
    [30, 15, 250e3, 20e-9, 0.1],
    [20, 25, 250e3, 20e-9, 0.2]
]

total_start_time = time.time()

for i, val_test in enumerate(val_tests):
    start_time = time.time()
    
    L = val_test[0] *1e-6
    C = val_test[1] * 1e-6
    fsw = val_test[2]
    t_d = val_test[3]
    d_cycle = val_test[4]

    DCR = calculate_dcr(L, data_inductor, d, rho) *1e-3
    ESR = calculate_esr(tan_delta, fsw, C)

    # Set Plecs parameters and simulate
    plecs.set(model+'/FETD1', 'Ron', str(R_ON_H))
    plecs.set(model+'/FETD2', 'Ron', str(R_ON_L))
    plecs.set(model+'/L1', 'L', str(L))
    plecs.set(model+'/DCR', 'R', str(DCR))
    plecs.set(model+'/ESR', 'R', str(ESR))
    plecs.set(model+'/C', 'C', str(C))
    plecs.set(model+'/Symmetrical PWM', 'fc', str(fsw))
    plecs.set(model+'/Deadtime', 'td', str(t_d))
    plecs.set(model+'/Duty Cycle', 'Value', str(d_cycle))
    times = plecs.simulate(model)['Time']
    current = plecs.simulate(model)['Values'][0]
    voltage = plecs.simulate(model)['Values'][1]

    # Calculate average and delta current using the function
    average_current, delta_current, delta_current_percentage = calculate_current(times, current, start_time=0.000, end_time=0.005)
    average_voltage, delta_voltage, delta_voltage_percentage = calculate_voltage(times, voltage, start_time=0.000, end_time=0.005)
    P_ON_H, P_ON_L = calculate_conduction_loss(average_current, average_voltage, V_in, R_ON_H, R_ON_L)
    P_COSS = calculate_output_capacitance_loss(V_in, fsw, C_OSS_H)
    P_D = calculate_dead_time_loss(V_D, average_current, fsw, t_d)
    P_G = calculate_gate_charge_loss(fsw, Q_g_H, Q_g_L, V_GS)
    P_L_DCR = calculate_inductor_conduction_loss(average_current, DCR)
    P_CAP_ESR = calculate_capacitor_loss(delta_current, ESR)
    P_total = calculate_total_losses(P_ON_H, P_ON_L, P_COSS, P_D, P_G, P_L_DCR, P_CAP_ESR)

    simulation_duration = time.time() - start_time

    print(f"Test {i+1}:")
    print(f"Average Current between 0.004 and 0.005 seconds: {average_current} A")
    print(f"Difference between highest and lowest current values between 0.004 and 0.005 seconds: {delta_current} A")
    print(f"Difference between highest and lowest current values between 0.004 and 0.005 seconds: {delta_current_percentage} %")
    print(f"Difference between highest and lowest voltage values between 0.004 and 0.005 seconds: {delta_voltage} V")
    print(f"Difference between highest and lowest voltage values between 0.004 and 0.005 seconds: {delta_voltage_percentage} %")
    print(f"Conduction Loss: {P_ON_H} and {P_ON_L} W")
    print(f"Output Capacitance Loss: {P_COSS} W")
    print(f"Dead Time Loss: {P_D} W")
    print(f"Gate Charge Loss: {P_G} W")
    print(f"Inductor Conduction Loss: {P_L_DCR} W")
    print(f"Capacitor Loss: {P_CAP_ESR} W")
    print(f"Total Losses: {P_total} W")
    print(f"Simulation Duration: {simulation_duration:.2f} seconds")
    print("-" * 50)

total_duration = time.time() - total_start_time
print(f"Total Simulation Duration: {total_duration:.2f} seconds")
                                                                                                                                                                                                                                                                                                                                                                                            


# # Loop

# In[96]:


# Loop through the combinations and simulate
csv_data = []
total_simulation_duration = 0  # Initialize total simulation duration

for simulation_num, val_test in enumerate(combinations, start=1):
    start_time = time.time()
    print(f"Starting Simulation {simulation_num}/{len(combinations)}")
    L = val_test[0] * 1e-6  # Convert to Henries
    C = val_test[1] * 1e-6  # Convert to Farads
    fsw = val_test[2]
    t_d = val_test[3]
    d_cycle = val_test[4]

    L_str = f"{L:.15f}"
    C_str = f"{C:.15f}"
    
    # print(f"Starting Simulation {simulation_num}")
    # print(f"Initial Values - L: {L}, C: {C}, fsw: {fsw}, t_dt: {t_d}, d_cycle: {d_cycle}")

    # print(f"After Initial Values - L: {L_str}, C: {C_str}, fsw: {fsw}, t_dt: {t_d}, d_cycle: {d_cycle}")

    DCR = calculate_dcr(L, data_inductor, d, rho) * 1e-3  # Convert to ohms
    ESR = calculate_esr(tan_delta, fsw, C)
    DCR_str = f"{DCR:.15f}"
    ESR_str = f"{ESR:.15f}"

    # print(f"Calculated DCR: {DCR}, ESR: {ESR}")
    # print(f"Calculated DCR: {DCR_str}, ESR: {ESR_str}")

    plecs.set(model+'/FETD1', 'Ron', str(R_ON_H))
    plecs.set(model+'/FETD2', 'Ron', str(R_ON_L))
    plecs.set(model+'/L1', 'L', L_str)
    plecs.set(model+'/DCR', 'R', DCR_str)
    plecs.set(model+'/ESR', 'R', ESR_str)
    plecs.set(model+'/C', 'C', C_str)
    plecs.set(model+'/Symmetrical PWM', 'fc', str(fsw))
    plecs.set(model+'/Deadtime', 'td', str(t_d))
    plecs.set(model+'/Duty Cycle', 'Value', str(d_cycle))

    simulation_results = plecs.simulate(model)
    times = simulation_results['Time']
    current = simulation_results['Values'][0]
    voltage = simulation_results['Values'][1]

    average_current, delta_current, delta_current_percentage = calculate_current(times, current, start_time=0.004, end_time=0.005)
    average_voltage, delta_voltage, delta_voltage_percentage = calculate_voltage(times, voltage, start_time=0.004, end_time=0.005)
    P_ON_H, P_ON_L = calculate_conduction_loss(average_current, average_voltage, V_in, R_ON_H, R_ON_L)
    P_COSS = calculate_output_capacitance_loss(V_in, fsw, C_OSS_H)
    P_D = calculate_dead_time_loss(V_D, average_current, fsw, t_d)
    P_G = calculate_gate_charge_loss(fsw, Q_g_H, Q_g_L, V_GS)
    P_L_DCR = calculate_inductor_conduction_loss(average_current, DCR)
    P_CAP_ESR = calculate_capacitor_loss(delta_current, ESR)
    P_total = calculate_total_losses(P_ON_H, P_ON_L, P_COSS, P_D, P_G, P_L_DCR, P_CAP_ESR)

    simulation_duration = time.time() - start_time
    total_simulation_duration += simulation_duration  # Add to total simulation duration

    csv_data.append([simulation_num, L_str, C_str, fsw, t_d, d_cycle, average_current, delta_current, delta_current_percentage, average_voltage, delta_voltage, delta_voltage_percentage, DCR_str, ESR_str, P_ON_H, P_ON_L, P_COSS, P_L_DCR, P_D, P_G, P_CAP_ESR, P_total])

    # print(f"Test {simulation_num}:")
    # print(f"Average Current between 0.004 and 0.005 seconds: {average_current} A")
    # print(f"Difference between highest and lowest current values between 0.004 and 0.005 seconds: {delta_current} A")
    # print(f"Difference between highest and lowest current values between 0.004 and 0.005 seconds: {delta_current_percentage} %")
    # print(f"Average Voltage between 0.004 and 0.005 seconds: {average_voltage} V")
    # print(f"Difference between highest and lowest voltage values between 0.004 and 0.005 seconds: {delta_voltage} V")
    # print(f"Difference between highest and lowest voltage values between 0.004 and 0.005 seconds: {delta_voltage_percentage} %")
    # print(f"Conduction Loss: {P_ON_H} and {P_ON_L} W")
    # print(f"Output Capacitance Loss: {P_COSS} W")
    # print(f"Inductor Conduction Loss: {P_L_DCR} W")
    # print(f"Dead Time Loss: {P_D} W")
    # print(f"Gate Charge Loss: {P_G} W")
    # print(f"Capacitor Loss: {P_CAP_ESR} W")
    # print(f"Total Losses: {P_total} W")
    # print(f"Simulation Duration: {simulation_duration:.2f} seconds")
    # print("-" * 50)

print(f"Total Simulation Duration: {total_simulation_duration:.2f} seconds")


# In[97]:


# Use the function to write data to the CSV file
write_to_csv(csv_file_path, csv_header, csv_data)

