import numpy as np
import pandas as pd


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

    delta_voltage_percentage = (delta_voltage/average_voltage) * 100

    return average_voltage, delta_voltage, delta_voltage_percentage

def calculate_current_1(current):

    # Calculate the average current value
    average_current = np.mean(current)

    # Calculate the difference between the highest and lowest current values
    delta_current = np.max(current) - np.min(current)

    # Calculate the percentage difference
    delta_current_percentage = (delta_current / average_current) * 100

    return average_current, delta_current, delta_current_percentage


def calculate_voltage_1(voltage):

    # Calculate the average voltage value
    average_voltage = np.mean(voltage)

    # Calculate the difference between the highest and lowest voltage values
    delta_voltage = np.max(voltage) - np.min(voltage)

    delta_voltage_percentage = (delta_voltage/average_voltage) * 100

    return average_voltage, delta_voltage, delta_voltage_percentage

def calculate_conduction_loss(I_out, V_out, V_IN, R_ON_H, R_ON_L):
    # Calculate P_ON-H
    P_ON_H = I_out**2 * R_ON_H * (V_out / V_IN)
    
    # Calculate P_ON-L
    P_ON_L = I_out**2 * R_ON_L * (1 - (V_out / V_IN))
    
    return P_ON_H, P_ON_L

def calculate_output_capacitance_loss(V_IN, f_sw, C_OSS_H):
    # Calculate P_COSS
    P_COSS = 0.5 * C_OSS_H * V_IN**2 * f_sw

    return P_COSS

def calculate_dead_time_loss(V_D, I_out, f_sw, t_d):

    # Calculate P_D
    P_D = V_D * I_out * t_d * f_sw

    return P_D

def calculate_gate_charge_loss(f_sw, Q_g_H, Q_g_L, V_GS):

    # Calculate P_G
    P_G = (Q_g_H + Q_g_L) * V_GS * f_sw

    return P_G

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

def calculate_total_losses(P_ON_H, P_ON_L, P_COSS, P_D, P_G, P_L_DCR, P_CAP_ESR):
    P_total = P_ON_H + P_ON_L  + P_COSS + P_D + P_G + P_L_DCR + P_CAP_ESR
    return P_total