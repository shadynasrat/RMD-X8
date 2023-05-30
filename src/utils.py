import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import scipy.signal as signal
import math
import csv
import pickle
from scipy.signal import chirp, spectrogram, gausspulse, sweep_poly
from scipy.signal import butter, filtfilt

def PKL_2_CSV():
    data = pickle.load(open("/home/sj/Desktop/SHADYcodes/RMD-X8/logs/others/log.pkl","rb"))

    datas = data['hardware_closed_loop'][1]

    tau_ests = np.zeros((len(datas), 12))
    torques = np.zeros((len(datas), 12))
    joint_positions = np.zeros((len(datas), 12))
    joint_position_targets = np.zeros((len(datas), 12))
    joint_velocities = np.zeros((len(datas), 12))

    for i in range(len(datas)):
        tau_ests[i, :] = datas[i]["tau_est"]
        torques[i, :] = datas[i]["torques"]
        joint_positions[i, :] = datas[i]["joint_pos"]
        joint_position_targets[i, :] = datas[i]["joint_pos_target"]
        joint_velocities[i, :] = datas[i]["joint_vel"]

    with open('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/others/original_file.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['MulPOS_1', 'GoalPOS_1', 'VEL_1', 'INTorque_1', 'MulPOS_2', 'GoalPOS_2', 'VEL_2', 'INTorque_2'])
        
        for i in range(len(joint_position_targets)):
            row = [joint_positions[i,1], joint_position_targets[i,1], joint_velocities[i,1], tau_ests[i,1], 
                   joint_positions[i,3], joint_position_targets[i,3], joint_velocities[i,3], tau_ests[i,3]]
            writer.writerow(row)

def merge_2_datasets():

    log_dir = '/home/sj/Desktop/SHADYcodes/RMD-X8/logs/'
    file_1 = 'bigboyjump_nvrs.csv'
    file_2 = 'aircircles.csv'
    file_M = 'bigboyMix.csv'

    df1 = pd.read_csv(log_dir+file_1)
    df2 = pd.read_csv(log_dir+file_2)
    merged_df = pd.concat([df1, df2], axis=1)
    merged_df.to_csv(log_dir+file_M, index=False)

    with open(log_dir+file_1, 'r') as f1, open(log_dir+file_2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        # Create a new output file
        with open(log_dir+file_M, 'w', newline='') as output_file:
            writer = csv.writer(output_file)

            # Write the header row to the output file
            header = next(reader1)
            writer.writerow(header)

            # Write the data rows from file 1 to the output file
            for row in reader1:
                writer.writerow(row)

            # Write the data rows from file 2 to the output file
            for row in reader2:
                writer.writerow(row)

def edit_dataset():
    df = pd.read_csv('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/data/200HZ/SweepStand.csv')
    df[['INTorque_1']] *= -1
    df[['TRQ_1']] *= -1
    df.to_csv('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/data/200HZ/SweepStand_shN.csv', index=True)

def signal_generator():

    """        am_modulation_index = 100
        carrier_freq = 1
        fm_modulation_index = 100

        t = time.time()
        signal = self.y
        am_modulated_wave = (1 + am_modulation_index * signal) * math.sin(2 * math.pi * carrier_freq * t)
        fm_modulated_wave = math.sin(2 * math.pi * (carrier_freq + fm_modulation_index * signal) * t)
        combined_wave = am_modulated_wave + fm_modulated_wave
    """
    # Define the parameters for the Gaussian pulse
    fc = 1/10000000  # carrier frequency (Hz)
    bw = 1  # bandwidth (Hz)
    fs = 100  # sampling frequency (Hz)
    duration = 1000  # duration of the pulse (s)
    t = np.linspace(0, 100000, num=duration)  # time vector (s)

    # Generate the Gaussian pulse
    gauss = gausspulse(t, fc=fc, bw=bw)

    # Define the parameters for the swept sine wave
    f0 = 1  # starting frequency (Hz)
    f1 = 20  # ending frequency (Hz)
    T = duration  # duration of the sweep (s)
    k = (f1 - f0) / T  # rate of frequency change (Hz/s)
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)  # phase angle (rad)

    # Generate the swept sine wave
    sweep = np.sin(phi)

    # Combine the two signals by element-wise multiplication
    combined = gauss * sweep

    # Plot the signals and the combined signal
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    axs[0].plot(t, gauss)
    axs[0].set_title('Gaussian pulse')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')

    axs[1].plot(t, sweep)
    axs[1].set_title('Swept sine wave')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')

    axs[2].plot(t, combined)
    axs[2].set_title('Combined signal')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def update_dataset_velocities():
    df = pd.read_csv('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/test.csv')
    velocities_1 = [0]
    velocities_2 = [0]
    for i in range(len(df['MulPOS_1'])-1):
        position1 = df['MulPOS_1'].iloc[i]
        position2 = df['MulPOS_1'].iloc[i+1]
        position3 = df['MulPOS_2'].iloc[i]
        position4 = df['MulPOS_2'].iloc[i+1]
        velocity_1 = (position2 - position1) / (1/1000)
        velocity_2 = (position4 - position3) / (1/1000)
        velocities_1.append(velocity_1)
        velocities_2.append(velocity_2)
    
    df['VEL_1_new'] = velocities_1
    df['VEL_2_new'] = velocities_2
    df.to_csv('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/test_vel.csv', index=True)

def down_sample_dataset():
    df = pd.read_csv('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/data/1000HZ/bigboyjump.csv')
    sample_to = 200

    n_samples_original = len(df)
    n_samples_resampled = int(n_samples_original * sample_to/1000)
    index_original = [int(i * n_samples_original / n_samples_resampled) for i in range(n_samples_resampled)]
    df_resampled = df.iloc[index_original]
    df_resampled.to_csv('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/data/sampled/bigboyjump_{sample_to}HZ.csv'.format(sample_to=sample_to))
    print(len(df), len(df_resampled))

def dataset_command_2_torque():
    df = pd.read_csv('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/data/bigboyjump_command.csv')
    df[['INTorque_1']] = df[['INTorque_1']] / 62.5 / 0.7
    df[['TRQ_1']] *= df[['TRQ_1']] / 62.5 / 0.7
    df[['INTorque_2']] *= -43.27201106093806
    df[['TRQ_2']] *= -43.27201106093806
    df.to_csv('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/data/bigboyjump_torque_shoulderonly.csv', index=True)


class plot():
    def smooth_n_plot():
        data = pd.read_csv('/home/sj/Desktop/SHADYcodes/RMD-X8/logs/data/200HZ/SweepStand.csv')
        sample_rate = 200
        nyquist_freq = 0.5 * sample_rate
        cutoff_freq = 10  # Hz
        order = 2

        # Calculate the filter coefficients
        normalized_cutoff_freq = cutoff_freq / nyquist_freq
        b, a = butter(order, normalized_cutoff_freq, btype='low')

        # Apply the filter to the data
        data['VEL_1_smooth'] = filtfilt(b, a, data['VEL_1'])
        data['VEL_2_smooth'] = filtfilt(b, a, data['VEL_2'])
        data['INTorque_1_smooth'] = filtfilt(b, a, data['INTorque_1'])
        data['INTorque_2_smooth'] = filtfilt(b, a, data['INTorque_2'])

        sh = data[['VEL_1', 'MulPOS_1', 'GoalPOS_1', 'INTorque_1', 'VEL_1_smooth', 'INTorque_1_smooth']]
        ne = data[['VEL_2', 'MulPOS_2', 'GoalPOS_2', 'INTorque_2', 'VEL_2_smooth', 'INTorque_2_smooth']]

        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
        axs[0].plot(sh.values)
        axs[0].plot(sh.values)
        axs[0].set_title('Walk-These-Ways Shoulder')
        axs[0].legend(sh.columns)  # add legend with column names
        axs[1].plot(ne.values)
        axs[1].set_title('Walk-These-Ways Knee')
        axs[1].legend(ne.columns)  # add legend with column names

        plt.tight_layout(pad=3.0)
        plt.show()

    def plot_comparison():
        """
        comparision between 2 datasets
        """
        df1 = pd.read_csv(filepath_or_buffer="/home/sj/Desktop/SHADYcodes/RMD-X8/logs/others/original_file.csv")
        df2 = pd.read_csv(filepath_or_buffer="/home/sj/Desktop/SHADYcodes/RMD-X8/logs/data/200HZ/SweepStandNF.csv")

        others_sh = df1[['VEL_1', 'MulPOS_1', 'GoalPOS_1', 'INTorque_1']]
        others_ne = df1[['VEL_2', 'MulPOS_2', 'GoalPOS_2', 'INTorque_2']]
        solo_sh = df2[['VEL_1', 'MulPOS_1', 'GoalPOS_1', 'INTorque_1']]
        solo_ne = df2[['VEL_2', 'MulPOS_2', 'GoalPOS_2', 'INTorque_2']]

        # my_torque['INTorque_2'] = my_torque['INTorque_2'].rolling(130, min_periods=1).mean()
        # my_pos['VEL_2'] = my_pos['VEL_2'].rolling(130, min_periods=1).mean()



        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs[0, 0].plot(others_sh.values)
        axs[0, 0].set_title('Walk-These-Ways Shoulder')
        axs[0, 0].legend(others_sh.columns)  # add legend with column names
        axs[1, 0].plot(others_ne.values)
        axs[1, 0].set_title('Walk-These-Ways Knee')
        axs[1, 0].legend(others_ne.columns)  # add legend with column names

        axs[0, 1].plot(solo_sh.values)
        axs[0, 1].set_title('SOLO Shoulder')
        axs[0, 1].legend(solo_sh.columns)  # add legend with column names
        axs[1, 1].plot(solo_ne.values)
        axs[1, 1].set_title('SOLO Knee')
        axs[1, 1].legend(solo_ne.columns)  # add legend with column names

        plt.tight_layout(pad=3.0)
        plt.show()

    def plot_dataset():
        """
        plot dataset showing knee and shoulder data
        """
        df = pd.read_csv(filepath_or_buffer="/home/sj/Desktop/SHADYcodes/RMD-X8/logs/bigboyjump_nvrs_400HZ.csv")
        sh = df[['VEL_1', 'MulPOS_1', 'GoalPOS_1', 'INTorque_1']]
        ne = df[['VEL_2', 'MulPOS_2', 'GoalPOS_2', 'INTorque_2']]

        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
        axs[0].plot(sh.values)
        axs[0].set_title('Walk-These-Ways Shoulder')
        axs[0].legend(sh.columns)  # add legend with column names
        axs[1].plot(ne.values)
        axs[1].set_title('Walk-These-Ways Knee')
        axs[1].legend(ne.columns)  # add legend with column names

        plt.tight_layout(pad=3.0)
        plt.show()





if __name__ == "__main__":
    down_sample_dataset()
    
