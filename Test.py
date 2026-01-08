### Lab02 Navigationssysteme - Pedestrian Dead Reckoning ###

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path

# ---------------------------
# PARAMETERS
# ---------------------------
params = {
    'sg_window': 21,          # Fenstergröße für Savitzky-Golay
    'sg_polyorder': 3,        # Polynomgrad
    'step_height_factor': 0.5, # adaptive Peak-Höhe
    'min_distance_s': 0.5,    # min Abstand zwischen Peaks (s)
    'weinberg_k': 0.4,        # Weinberg-Faktor für Schrittlänge
    'stair_threshold': 0.15,  # Höhenänderung für Treppen (m)
    'start_lat': 47.06422,
    'start_lon': 15.45291,
}

# ---------------------------
# FUNCTIONS
# ---------------------------

def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def compute_total_acceleration(acc_df):
    acc_df['acc_total'] = np.sqrt(acc_df['acc_x']**2 + acc_df['acc_y']**2 + acc_df['acc_z']**2)
    return acc_df

def apply_savgol(signal, window_length, polyorder):
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)

def detect_steps(acc_signal, fs, height_factor, min_distance_s):
    # adaptive Schwelle
    height_threshold = np.mean(acc_signal) + height_factor * np.std(acc_signal)
    distance_samples = int(min_distance_s * fs)
    peaks, _ = find_peaks(acc_signal, height=height_threshold, distance=distance_samples)
    return peaks

def compute_step_lengths(acc_signal, peaks, k):
    # Talwerte (Valleys) zwischen Peaks
    acc_valleys = [np.min(acc_signal[:peaks[0]+1])]
    for i in range(1, len(peaks)):
        acc_valleys.append(np.min(acc_signal[peaks[i-1]:peaks[i]+1]))
    acc_valleys = np.array(acc_valleys)
    
    # Weinberg-Modell
    step_lengths = k * (acc_signal[peaks] - acc_valleys)**0.25
    return step_lengths, acc_valleys

def adjust_step_lengths_with_height(step_lengths, height_at_peaks, stair_threshold, stair_length=0.3):
    delta_h = np.diff(height_at_peaks, prepend=height_at_peaks[0])
    adjusted_lengths = np.where(np.abs(delta_h) > stair_threshold, stair_length, step_lengths)
    return adjusted_lengths

def estimate_trajectory(step_lengths, yaw_steps, lat0, lon0):
    R = 6378137  # Erdradius in m
    N = np.zeros(len(step_lengths) + 1)
    E = np.zeros(len(step_lengths) + 1)
    
    for i in range(len(step_lengths)):
        N[i+1] = N[i] + step_lengths[i] * np.cos(yaw_steps[i])
        E[i+1] = E[i] + step_lengths[i] * np.sin(yaw_steps[i])
    
    lat = lat0 + (N / R) * (180 / np.pi)
    lon = lon0 + (E / (R * np.cos(np.deg2rad(lat0)))) * (180 / np.pi)
    
    return N, E, lat, lon

# ---------------------------
# DATA IMPORT
# ---------------------------
dir = Path(__file__).resolve().parent
folder = Path(dir)/"Daten"/"DatenRaw"

acc = pd.read_csv(folder/'accelerometer.csv')
gyro = pd.read_csv(folder/'gyroscope.csv')
mag = pd.read_csv(folder/'magnetometer.csv')
baro = pd.read_csv(folder/'barometer.csv')
ori = pd.read_csv(folder/'referenceOrientation.csv')
ground_truth = pd.read_csv(folder/'groundTruth.csv')

# Zeit in Sekunden
acc['time'] = acc['time']/1000
baro['time'] = baro['time']/1000
ori['time'] = ori['time']/1000

fs = 1/np.mean(np.diff(acc['time']))  # Samplingrate schätzen

# ---------------------------
# STEP 1: TOTAL ACC
# ---------------------------
acc = compute_total_acceleration(acc)
acc['acc_filtered'] = apply_savgol(acc['acc_total'], params['sg_window'], params['sg_polyorder'])

# ---------------------------
# STEP 2: STEP DETECTION
# ---------------------------
peaks = detect_steps(acc['acc_filtered'], fs, params['step_height_factor'], params['min_distance_s'])
step_count = len(peaks)

print(f"Detected steps: {step_count}")

# ---------------------------
# STEP 3: HEIGHT & STEP LENGTHS
# ---------------------------
# Barometrische Höhe
p0 = baro['pressure'].iloc[0]
baro['height'] = (293.15 / 6.5e-3) * ((baro['pressure']/p0)**(-0.1903) - 1)
baro['height_filtered'] = apply_savgol(baro['height'], params['sg_window'], params['sg_polyorder'])

# Schrittweiten
step_lengths, acc_valleys = compute_step_lengths(acc['acc_filtered'].values, peaks, params['weinberg_k'])

# Höhenanpassung
height_at_peaks = np.interp(acc['time'].iloc[peaks], baro['time'], baro['height_filtered'])
step_lengths_adjusted = adjust_step_lengths_with_height(step_lengths, height_at_peaks, params['stair_threshold'])

# ---------------------------
# STEP 4: YAW ESTIMATION
# ---------------------------
roll = np.arctan2(-acc['acc_x'], -acc['acc_z'])
pitch = np.arctan2(acc['acc_y'], np.sqrt(acc['acc_x']**2 + acc['acc_z']**2))

yaw = np.arctan2(
    -mag['mag_x']*np.cos(roll) + mag['mag_z']*np.sin(roll),
     mag['mag_y']*np.cos(pitch) + mag['mag_x']*np.sin(pitch)*np.sin(roll) + mag['mag_z']*np.sin(pitch)*np.cos(roll)
)
yaw = (yaw + np.pi) % (2*np.pi) - np.pi
yaw_filtered = apply_savgol(yaw, params['sg_window'], params['sg_polyorder'])
yaw_steps = yaw_filtered[peaks]

# ---------------------------
# STEP 5: TRAJECTORY
# ---------------------------
N, E, lat, lon = estimate_trajectory(step_lengths_adjusted, yaw_steps, params['start_lat'], params['start_lon'])

# Ergebnis speichern
df_geo = pd.DataFrame({
    'step_id': np.arange(len(lat)),
    'time_s': np.insert(acc['time'].iloc[peaks].values, 0, acc['time'].iloc[peaks].values[0]),
    'latitude': lat,
    'longitude': lon,
    'north_m': N,
    'east_m': E
})
df_geo.to_csv("pdr_trajectory_full.csv", index=False)

# ---------------------------
# STEP 6: ENDPOINT ERROR
# ---------------------------
N_pdr_end, E_pdr_end = N[-1], E[-1]
N_gt_end, E_gt_end = ground_truth['y'].values[-1], ground_truth['x'].values[-1]

endpoint_error = np.sqrt((N_pdr_end - N_gt_end)**2 + (E_pdr_end - E_gt_end)**2)
print(f"Endpoint error: {endpoint_error:.2f} m")

# ---------------------------
# PLOTTING
# ---------------------------
plt.figure(figsize=(10,4))
plt.plot(acc['time'], acc['acc_total'], alpha=0.4, label='Raw acceleration')
plt.plot(acc['time'], acc['acc_filtered'], label='Filtered acceleration')
plt.plot(acc['time'].iloc[peaks], acc['acc_filtered'].iloc[peaks], "x", label='Detected steps')
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title("Step Detection")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(E, N, label='PDR trajectory')
plt.plot(ground_truth['x'], ground_truth['y'], label='Ground Truth')
plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.title("Trajectory Comparison")
plt.legend()
plt.grid(True)
plt.show()
