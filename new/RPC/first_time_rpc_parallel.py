import xmlrpc.client
import jsonrpc_requests  # for JSON-RPC
import collections.abc  # to make jsonrpc_requests usable for Python 3.10+
import os
import numpy as np
import time  # Import time module for measuring execution time

# Define constants and parameters
HOST_ADDRESS = "http://localhost:1080/RPC2"
MODEL_NAME = "buck_new"
METHOD = "JSON"  # "XML", "JSON"

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

# Import RPC module
if METHOD == "JSON":
    collections.Mapping = collections.abc.Mapping
    server = jsonrpc_requests.Server(HOST_ADDRESS)
elif METHOD == "XML":
    server = xmlrpc.client.Server(HOST_ADDRESS)

# Get the current working directory and set the full path for the model
current_directory = os.getcwd()
model_path = os.path.join(current_directory, MODEL_NAME)

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
    DCR = 1e-3
    ESR = 1e-3

    opts = {
        'ModelVars': {
            'varL': L,
            'varC': C,
            'varFsw': fsw,
            'varTd': t_d,
            'varDutyCycle': d_cycle,
            'varDCR': DCR,
            'varESR': ESR
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

            average_current, delta_current, delta_current_percentage = calculate_current(times, current, start_time=0.004, end_time=0.005)
            
            execution_time = end_time - start_time

            results.append({
                'average_current': average_current,
                'delta_current': delta_current,
                'delta_current_percentage': delta_current_percentage,
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
        print(f"  Execution Time: {result['execution_time']} seconds")

# Print the total execution time for all simulations
print(f"Total execution time for all simulations: {total_execution_time} seconds")
