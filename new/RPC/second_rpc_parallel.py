import xmlrpc.client
import jsonrpc_requests  # for JSON-RPC
import collections.abc  # to make jsonrpc_requests usable for Python 3.10+
import os
import numpy as np
import time  # Import time module for measuring execution time
from calc import *
import pandas as pd

# Define constants and parameters
HOST_ADDRESS = "http://localhost:1080/RPC2"
MODEL_NAME = "rpc_buck_new"
METHOD = "JSON"  # "XML", "JSON"

# Import RPC module
if METHOD == "JSON":
    collections.Mapping = collections.abc.Mapping
    server = jsonrpc_requests.Server(HOST_ADDRESS)
elif METHOD == "XML":
    server = xmlrpc.client.Server(HOST_ADDRESS)

# Get the current working directory and set the full path for the model
current_directory = os.getcwd()
model_path = os.path.join(current_directory, MODEL_NAME)

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
num_values = 2          # Number of values for each parameter

# Load lookup tables
inductor_lookup_table = r'E:\ai-power-converter-1\new\dataset\lookup_inductor.csv'
capacitor_lookup_table = r'E:\ai-power-converter-1\new\dataset\lookup_capacitor.csv'
data_inductor = pd.read_csv(inductor_lookup_table)

# Set up simulation parameters
val_tests = [
    [30e-6, 15e-6, 250e3, 20e-9, 0.1],
    [30e-6, 15e-6, 250e3, 20e-9, 0.2]
]

# Define the start and end times for the output
start_time = 0.0
end_time = 1e-2
step_size = 2e-8

output_times_vec = np.arange(start_time, end_time, step_size).tolist()

simStructs = []
for val_test in val_tests:
    L, C, fsw, t_d, d_cycle = val_test
    DCR = calculate_dcr(L, data_inductor, d, rho)
    ESR = calculate_esr(tan_delta, fsw, C)

    opts = {
        'ModelVars': {
            'varRonH': R_ON_H,
            'varRonL': R_ON_L,
            'varL': L,
            'varDCR': DCR,
            'varESR': ESR,
            'varC': C,
            'varFsw': fsw,
            'varTd': t_d,
            'varDutyCycle': d_cycle,
        },
        'SolverOpts': {
            'OutputTimes': output_times_vec
        },
        'Name': f'L = {L} H, C = {C} F, fsw = {fsw} Hz, td = {t_d} s, Duty = {d_cycle}'
    }
    simStructs.append(opts)

callback = """
if ischar(result)
    disp(["There is a simulation error for the case (" name ")."]);
else
    plecs('scope', './Scope', 'HoldTrace', name);
    
    [maxi, maxidx] = max(result.Values(1,:));
    maxt = result.Time(maxidx);
    result = [maxi, maxt];
end
"""

# Function to run simulations and process results
def run_simulations(simStructs):
    results = []
    for opts in simStructs:
        try:
            start_time = time.time()  # Start time measurement
            result = server.plecs.simulate(MODEL_NAME, opts)
            end_time = time.time()  # End time measurement
            
            times = result['Time']
            current = result['Values'][0]
            voltage = result['Values'][1]

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
            
            execution_time = end_time - start_time

            results.append({
                'average_current': average_current,
                'delta_current': delta_current,
                'delta_current_percentage': delta_current_percentage,
                'average_voltage': average_voltage,
                'delta_voltage': delta_voltage,
                'delta_voltage_percentage': delta_voltage_percentage,
                'P_ON_H': P_ON_H,
                'P_ON_L': P_ON_L,
                'P_COSS': P_COSS,
                'P_D': P_D,
                'P_G': P_G,
                'P_L_DCR': P_L_DCR,
                'P_CAP_ESR': P_CAP_ESR,
                'P_total': P_total,
                'execution_time': execution_time
            })
        except Exception as e:
            results.append(str(e))
    return results

# Load model and clear traces
server.plecs.load(model_path)
server.plecs.scope(MODEL_NAME + "/Scope", "ClearTraces")

# Measure the total execution time for all simulations
total_start_time = time.time()  # Start total time measurement

# Run simulations and collect results
results = run_simulations(simStructs)

total_end_time = time.time()  # End total time measurement
total_execution_time = total_end_time - total_start_time

# Close the model
# server.plecs.close(MODEL_NAME)

# Process and print results
for i, result in enumerate(results):
    if isinstance(result, str):
        print(f"Simulation {i} failed with error: {result}")
    else:
        print(f"Simulation {i} results:")
        print(f"  Average Current: {result['average_current']} A")
        print(f"  Delta Current: {result['delta_current']} A ({result['delta_current_percentage']} %)")
        print(f"  Average Voltage: {result['average_voltage']} V")
        print(f"  Delta Voltage: {result['delta_voltage']} V ({result['delta_voltage_percentage']} %)")
        print(f"  P_ON_H: {result['P_ON_H']} W")
        print(f"  P_ON_L: {result['P_ON_L']} W")
        print(f"  P_COSS: {result['P_COSS']} W")
        print(f"  P_D: {result['P_D']} W")
        print(f"  P_G: {result['P_G']} W")
        print(f"  P_L_DCR: {result['P_L_DCR']} W")
        print(f"  P_CAP_ESR: {result['P_CAP_ESR']} W")
        print(f"  P_total: {result['P_total']} W")
        print(f"  Execution Time: {result['execution_time']} seconds")

# Print the total execution time for all simulations
print(f"Total execution time for all simulations: {total_execution_time} seconds")
