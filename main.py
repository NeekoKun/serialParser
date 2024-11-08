import matplotlib.pyplot as plt
import serial
import pandas as pd
import time
import numpy as np
import scipy.optimize

def read_serial_data():
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)
    while True:
        with open('data.csv', 'w') as f:
            try:
                ser.flush()
                msg = ser.readline()
                try:
                    value = msg.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                f.write(value)
            except serial.serialutil.SerialException:
                break

def parse_to_csv():
    with open('data.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    charge = [data.split(':') for data in lines[0].split(';')]
    discharge = [data.split(':') for data in lines[1].split(';')]

    charge_df = pd.DataFrame(charge, columns=['Parameter', 'Value'])
    discharge_df = pd.DataFrame(discharge, columns=['Parameter', 'Value'])
    charge_df.to_csv('data/charge.csv', index=False)
    discharge_df.to_csv('data/discharge.csv', index=False)

def plot_charge_and_discharge():
    # Reading Data

    charge_df = pd.read_csv('data/charge.csv')
    discharge_df = pd.read_csv('data/discharge.csv')

    charge_time = charge_df['Parameter'].astype(float)
    discharge_time = discharge_df['Parameter'].astype(float)
    time = pd.concat([charge_df['Parameter'], discharge_df['Parameter']]).astype(float)

    # Fitting Charge Data

    charge_values, _ = scipy.optimize.curve_fit(
        lambda t, a, b: a * (1 - np.exp(-t / b)),
        charge_time,
        charge_df['Value'].astype(float)
    )

    fitted_charge = charge_values[0] * (1 - np.exp(-charge_time / charge_values[1]))
    
    # Fitting Discharge Data

    discharge_values, _ = scipy.optimize.curve_fit(
        lambda t, a, b, c: a * np.exp(-(t + c) / b),
        discharge_time - discharge_time[0] + 1,
        discharge_df['Value'].astype(float)
    )

    fitted_discharge = discharge_values[0] * np.exp(-(discharge_time - discharge_time[0] + 1) / discharge_values[1])

    fitted_curve = list(fitted_charge) + list(fitted_discharge)

    # Getting Maximum Slopes

    slopes = get_maximum_slope()

    charge_slope_x = np.linspace(0, 2500, 100)
    charge_slope_y = slopes[0] * charge_slope_x

    discharge_slope_x = np.linspace(discharge_time[0], discharge_time[0] + 2500, 100)
    discharge_slope_y = slopes[1] * discharge_slope_x
    discharge_slope_y -= min(discharge_slope_y)

    # Expected Curve

    expected_charging = [expected_charge(i - charge_df['Parameter'][0]) for i in charge_time]
    expected_discharging = [expected_charge(i - discharge_df['Parameter'][0], False, 4.9) for i in discharge_time]
    expected_curve = expected_charging + expected_discharging

    ############
    # Plotting #
    ############

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Charge Data & Fitted Line

    axs[0, 0].scatter(charge_time, charge_df['Value'].astype(float), marker='o', s=10, label='Charge Data')
    axs[0, 0].plot(charge_time, fitted_charge, color='red', label='Fitted Line')
    axs[0, 0].fill_between(charge_time, charge_df['Value'] + 0.1, charge_df['Value'] - 0.1, color='blue', alpha=0.5, label='Error')
    axs[0, 0].plot(charge_slope_x, charge_slope_y, color='green', linestyle='--', label='Maximum Slope')
    axs[0, 0].legend()
    axs[0, 0].set_title('Charge Data & Fitted Line')
    axs[0, 0].set_xlabel('Milliseconds')
    axs[0, 0].set_ylabel('Tension')
    axs[0, 0].grid(True)
    axs[0, 0].tick_params(axis='x', rotation=45)

    # Discharge Data & Fitted Line

    axs[0, 1].scatter(discharge_time, discharge_df['Value'].astype(float), marker='o', s=10, label='Discharge Data')
    axs[0, 1].plot(discharge_time, fitted_discharge, color='red', label='Fitted Line')
    axs[0, 1].fill_between(discharge_time, discharge_df['Value'] + 0.1, discharge_df['Value'] - 0.1, color='blue', alpha=0.5, label='Error')
    axs[0, 1].plot(discharge_slope_x, discharge_slope_y, color='green', linestyle='--', label='Maximum Slope')
    axs[0, 1].set_title('Discharge Data & Fitted Line')
    axs[0, 1].set_xlabel('Milliseconds')
    axs[0, 1].set_ylabel('Tension')
    axs[0, 1].grid(True)
    axs[0, 1].tick_params(axis='x', rotation=45)

    # Expected Curve & Fitted Curve

    axs[1, 0].plot(time, expected_curve, marker=',', label='Expected Curve')
    axs[1, 0].plot(time, fitted_curve, color='blue', linestyle='--', label='Fitted Curve')
    axs[1, 0].legend()
    axs[1, 0].set_title('Fitted Curve & Expected Curve')
    axs[1, 0].set_xlabel('Milliseconds')
    axs[1, 0].set_ylabel('Tension')
    axs[1, 0].grid(True)
    axs[1, 0].tick_params(axis='x', rotation=45)

    # Logarithmic Charge Fitted Curve

    axs[1, 1].plot(discharge_time, discharge_df['Value'].astype(float), marker='o')
    axs[1, 1].set_title('Logarithmic Discharge Fitted Curve')
    axs[1, 1].set_xlabel('Milliseconds')
    axs[1, 1].set_ylabel('Tension')
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True)
    axs[1, 1].tick_params(axis='x', rotation=45)

    # Show Plot

    plt.tight_layout()
    plt.show()

def expected_charge(time: int, charging: bool = True, starting_charge: int = 0):
    v_max = 5
    tao = 2.7
    if charging:
        return v_max * (1 - np.exp(-(time / 1000) / tao))
    else:
        return starting_charge * np.exp(-(time / 1000) / tao)

def get_maximum_slope() -> list[float]:
    charge_df = pd.read_csv('data/charge.csv')
    charge_df['Value'] = charge_df['Value'].astype(float)
    charge_slopes = [(charge_df['Value'][i] - charge_df['Value'][i - 1]) / (charge_df['Parameter'][i] - charge_df['Parameter'][i - 1]) for i in range(1, len(charge_df['Value']))]
    max_slope = max(charge_slopes)

    discharge_df = pd.read_csv('data/discharge.csv')
    discharge_df['Value'] = discharge_df['Value'].astype(float)
    discharge_slopes = [(discharge_df['Value'][i] - discharge_df['Value'][i - 1]) / (discharge_df['Parameter'][i] - discharge_df['Parameter'][i - 1]) for i in range(1, len(discharge_df['Value']))]
    min_slope = min(discharge_slopes)

    return max_slope, min_slope

plot_charge_and_discharge()
