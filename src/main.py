from rmd_x8 import RMD_X8
import time
import pandas as pd
import numpy as np
from math import pi, sin, floor
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import keyboard
import torch
import os
import sys
import traceback
import statistics
from kinamatics import SOLO_kinematics
import threading
import scipy.signal as signal

#from PyViveTracker.main import trackerData

shoulder_MTR = RMD_X8(0x0141)
shoulder_MTR.setup("4")
knee_MTR = RMD_X8(0x0142)
knee_MTR.setup("4")

IK = SOLO_kinematics() 
thread1 = threading.Thread(target=IK.render)
thread1.start()

csvData = []
csvData_knee = []
csvData_shoulder = []

start_time = time.time()
time_elapsed = 0

INIT_PDI_START = np.array([1,1,5], dtype='f')
INIT_PDI_FINAL = np.array([17,1.2,15], dtype='f')
JUMP_PDI_START = np.array([5,1,5], dtype='f')
JUMP_PDI_FINAL = np.array([17,0.58,5], dtype='f')

NUM_STEPS = 500000
DT = 0.001
INIT_GIAN_INCRMNT = (INIT_PDI_FINAL - INIT_PDI_START) / NUM_STEPS
JUMP_GIAN_INCRMNT = (JUMP_PDI_FINAL - JUMP_PDI_START) / NUM_STEPS
FREQ_INC = (3 - 0.1) / NUM_STEPS

TORQUE_RANGE = [-300,300]
KNEE_RANGE = [-19,90]
SHOULDER_RANGE = [-90,60]
SAFETY_OFFSET = 4
SELECT_MOTION = 1

Solo_Init = [-3,-18]
Solo_Jump = [66,-50]

if SELECT_MOTION == 1:
    Solo_Init = [-3,-18]
    Solo_Jump = [66,-50]
elif SELECT_MOTION == 1:
    Solo_Init = [-13,-12]
    Solo_Jump = [84,-57]

fs = 100                                                                    # sampling frequency
cutoff_freq = 5                                                             # cutoff frequency in Hz
order = 3                                                                   # filter order
nyquist_freq = 0.8 * fs
cutoff_norm = cutoff_freq / nyquist_freq                                    # normalized cutoff frequency
b, a = signal.butter(order, cutoff_norm, btype='lowpass')
shoulder_input_buffer = np.zeros(100)
shoulder_output_buffer = np.zeros(100)
knee_input_buffer = np.zeros(100)
knee_output_buffer = np.zeros(100)

def PDI_Gain_update():
    global INIT_PDI_START
    global JUMP_PDI_START
    
    INIT_PDI_START += INIT_GIAN_INCRMNT
    JUMP_PDI_START += JUMP_GIAN_INCRMNT

    return INIT_PDI_START, JUMP_PDI_START

def input_torque_filter(torque, torque_buf, filter_factor):
    filter_factor = 0.6
    torque_LMT = min(max(torque, TORQUE_RANGE[0]), TORQUE_RANGE[1])                     #Torque Limiter
    torque_LPF = torque_buf[0] * filter_factor + torque_LMT * (1 - filter_factor)       #Low Pass Filter
    torque_buf = np.roll(torque_buf, -1)
    torque_buf[0] = torque_LPF
    out_torque = sum(torque_buf) / len(torque_buf)                                      #Moving Average Filter
    return round(out_torque), output_buffer

def shoulder_apply_lowpass_filter(input_sample):
    shoulder_input_buffer[:-1] = shoulder_input_buffer[1:]
    shoulder_input_buffer[-1] = input_sample

    # Apply the filter to the input buffer
    shoulder_output_buffer[:-1] = shoulder_output_buffer[1:]
    shoulder_output_buffer[-1] = signal.lfilter(b, a, shoulder_input_buffer)[-1]

    # Return the latest filtered sample
    return round(shoulder_output_buffer[-1])

def knee_apply_lowpass_filter(input_sample):
    knee_input_buffer[:-1] = knee_input_buffer[1:]
    knee_input_buffer[-1] = input_sample

    # Apply the filter to the input buffer
    knee_output_buffer[:-1] = knee_output_buffer[1:]
    knee_output_buffer[-1] = signal.lfilter(b, a, knee_input_buffer)[-1]

    # Return the latest filtered sample
    return round(knee_output_buffer[-1])

def Solo_data_collection_using_kinematics_PID_control():
    start_time = time.time()
    shoulder_pos = Solo_Init[1]
    knee_pos = Solo_Init[0]
    shoulder_data_1 = {"MulPOS": -11}
    shoulder_data_2 = {"vel": 0}
    knee_data_1 = {"MulPOS": 10}
    knee_data_2 = {"vel": 0}
    knee_last_error = 0
    knee_integral = 0
    shoulder_last_error = 0
    shoulder_integral = 0
    time_elapsed = 0
    initialize(motorNum=motorNum)
    shoulder_torque = 0
    knee_torque = 0
    jump_count = 0
    time_elapsed_jump = 0
    last_time_elapsed_jump = 0
    jump_duration = 0
    knee_pos_last = 0
    knee_torque_buf = np.zeros(50, dtype=int)
    shoulder_torque_buf = np.zeros(50, dtype=int)
    shoulder_pos_last = 0
    filter_factor = 0.95
    p_gains = 5
    d_gains = 0.2
    i_gains = 15
    shoulder_current_pos = 0

    print("\nCheck initial pose, Starting in 1 seconds\n")
    time.sleep(1)
    IK.start = True
    print("\n-------PID Control Start-------\n")

    for i in range(NUM_STEPS):
        shoulder_data_2 = shoulder_MTR.torque_closed_loop(shoulder_torque)
        knee_data_2 = knee_MTR.torque_closed_loop(knee_torque)
        shoulder_data_1 = shoulder_MTR.read_multi_turns_angle()
        knee_data_1 = knee_MTR.read_multi_turns_angle()

       
        shoulder_current_pos = shoulder_data_1["MulPOS"]/900
        shoulder_error = shoulder_pos - shoulder_current_pos
        shoulder_integral += shoulder_error * DT
        shoulder_derivative = (shoulder_error - shoulder_last_error) / DT
        shoulder_last_error = shoulder_error
        shoulder_torque_PID = p_gains * shoulder_error + i_gains * shoulder_integral + d_gains * shoulder_derivative
        #shoulder_torque, shoulder_torque_buf = input_torque_filter(shoulder_torque_PID, shoulder_torque_buf, filter_factor)
        shoulder_torque = shoulder_apply_lowpass_filter(shoulder_torque_PID)

        knee_current_pos = knee_data_1["MulPOS"]/900
        knee_error = knee_pos - knee_data_1["MulPOS"]
        knee_integral += knee_error * DT
        knee_derivative = (knee_error - knee_last_error) / DT
        knee_last_error = knee_error
        knee_torque_PID = p_gains * knee_error + i_gains * knee_integral + d_gains * knee_derivative        
        #knee_torque, knee_torque_buf = input_torque_filter(knee_torque_PID, knee_torque_buf, filter_factor)
        knee_torque = knee_apply_lowpass_filter(knee_torque_PID)

        shoulder_pos, knee_pos = IK.theta1_offset, IK.theta2_offset
        IK.theta3, IK.theta4 = (shoulder_current_pos - 1) * math.pi / 180, (knee_current_pos - 150) * math.pi / 180

        last_time_elapsed = time_elapsed
        time_elapsed = time.time()
        t =  time_elapsed - start_time
        fps = 1 / (time_elapsed - last_time_elapsed)
        fps_data = {"fps": fps}

        shoulder_data = {"trq": shoulder_data_2["trq"], "INTorque": shoulder_torque, "GoalPOS": shoulder_pos * math.pi / 180, "vel": cal_velocity(shoulder_current_pos, shoulder_pos_last, time_elapsed - last_time_elapsed), "jump_duration": jump_duration, "MulPOS": shoulder_current_pos * math.pi / 180, "TMP": shoulder_data_2["TMP"]}
        shoulder_data_2.pop("pos")
        shoulder_data_1.pop("MulPOS")

        knee_data = {"trq": knee_data_2["trq"], "INTorque": knee_torque, "GoalPOS": knee_pos * math.pi / 180, "vel": cal_velocity(knee_current_pos, knee_pos_last, time_elapsed - last_time_elapsed), "jump_duration": jump_duration, "MulPOS": knee_current_pos * math.pi / 180, "TMP": knee_data_2["TMP"]}
        knee_data_2.pop("pos")
        knee_data_1.pop("MulPOS")

        jump_duration_data = {"jump_duration": jump_duration}

        csvData_shoulder.append({**shoulder_data})
        csvData_knee.append({**knee_data})
        
        print("step:{}\t{}℃  SHDR: {:.0f}°-->{:.0f}°\t torque: {:.0f}\t {}℃  KNEE: {:.0f}-->{:.0f}\t torque: {:.0f}\t JUMPS: {} Duration:{:.2f}\t\t TIME ELAPSED: {:.2f}\t FPS: {:.0f}\t PDI: {:.2f},{:.2f},{:.2f}".format(i, shoulder_data_2["TMP"], shoulder_current_pos, shoulder_pos, shoulder_torque, knee_data["TMP"], knee_current_pos, knee_pos, knee_torque, jump_count, jump_duration, t, fps, p_gains, d_gains, i_gains))
        
        if safety(knee_current_pos, shoulder_current_pos) == 1:
            break
    
    print("SHOULDER MAXSPEED: ",max(csvData_shoulder, key=lambda x: x['SPD'])['SPD'])
    print("KNEE     MAXSPEED: ",max(csvData_knee, key=lambda x: x['SPD'])['SPD'])
    fps = [fps_data['fps'] for data in csvData_shoulder]
    print("FPS              : ", round(statistics.median(fps)))
    #Jump = [jump_duration_data['jump_duration'] for i in jump_duration_data['jump_duration'][3:]]
    #print("Jump Duration    : ", statistics.median(Jump))
    Solo_plot_graph(csvData_shoulder, csvData_knee)

def Solo_data_collection_using_PID_control():
    start_time = time.time()
    shoulder_pos = Solo_Init[1]
    knee_pos = Solo_Init[0]
    shoulder_data_1 = {"MulPOS": 0}
    shoulder_data_2 = {"vel": 0}
    knee_data_1 = {"MulPOS": 0}
    knee_data_2 = {"vel": 0}
    p_gains = 20
    d_gains = 1
    i_gains = 15
    knee_last_error = 0
    knee_integral = 0
    shoulder_last_error = 0
    shoulder_integral = 0
    time_elapsed = 0
    initialize(motorNum=motorNum)
    shoulder_torque = 0
    knee_torque = 0
    jump_count = 0
    time_elapsed_jump = 0
    last_time_elapsed_jump = 0
    jump_duration = 0
    shoulder_torque_last = 0
    knee_torque_last = 0
    knee_pos_last = 0
    knee_torque_buf = []
    shoulder_torque_buf = np.zeros(10, dtype=int)
    shoulder_pos_last = 0
    filter_factor = 0.95

    print("\n-------PID Control Start-------\n")
    
    for i in range(NUM_STEPS):

        shoulder_data_2 = shoulder_MTR.torque_closed_loop(shoulder_torque)
        knee_data_2 = knee_MTR.torque_closed_loop(knee_torque)
        shoulder_data_1 = shoulder_MTR.read_multi_turns_angle()
        knee_data_1 = knee_MTR.read_multi_turns_angle()

        shoulder_data_1["MulPOS"] = shoulder_data_1["MulPOS"]/900
        shoulder_error = shoulder_pos - shoulder_data_1["MulPOS"]
        shoulder_integral += shoulder_error * DT
        shoulder_derivative = (shoulder_error - shoulder_last_error) / DT
        shoulder_last_error = shoulder_error
        shoulder_torque_PID = p_gains * shoulder_error + i_gains * shoulder_integral + d_gains * shoulder_derivative
        shoulder_torque_LMT = min(max(shoulder_torque_PID, TORQUE_RANGE[0]), TORQUE_RANGE[1])
        shoulder_torque_LPF = shoulder_torque_last * filter_factor + shoulder_torque_LMT * (1 - filter_factor)
        shoulder_torque_buf = np.roll(shoulder_torque_buf, -1)
        shoulder_torque_buf[0] = shoulder_torque_LPF
        shoulder_torque = round(sum(shoulder_torque_buf) / len(shoulder_torque_buf))
        shoulder_torque_last = shoulder_torque

        knee_data_1["MulPOS"] = knee_data_1["MulPOS"]/900
        knee_error = knee_pos - knee_data_1["MulPOS"]
        knee_integral += knee_error * DT
        knee_derivative = (knee_error - knee_last_error) / DT
        knee_last_error = knee_error
        knee_torque = p_gains * knee_error + i_gains * knee_integral + d_gains * knee_derivative            #THERE IS A PROBLEM HERE
        knee_torque = round(min(max(knee_torque, TORQUE_RANGE[0]), TORQUE_RANGE[1]))
        knee_torque = round(knee_torque_last * filter_factor + knee_torque * (1 - filter_factor))
        knee_torque_last = knee_torque

        last_time_elapsed = time_elapsed
        time_elapsed = time.time()
        t =  time_elapsed - start_time
        fps = 1 / (time_elapsed - last_time_elapsed)
        fps_data = {"fps": fps}

        if jump_count > 1:
            jump_duration = time_elapsed_jump - last_time_elapsed_jump

        shoulder_INPtorque = {"INTorque": shoulder_torque}
        shoulder_goalpos = {"GoalPOS": shoulder_pos}
        shoulder_data_2.pop("pos")
        shoulder_data_2["vel"] = cal_velocity(shoulder_data_1["MulPOS"], shoulder_pos_last, time_elapsed - last_time_elapsed)
        shoulder_pos_last = shoulder_data_1["MulPOS"]
        knee_INPtorque = {"INTorque": knee_torque}
        knee_goalpos = {"GoalPOS": knee_pos}
        knee_data_2.pop("pos")
        knee_data_2["vel"] = cal_velocity(knee_data_1["MulPOS"], knee_pos_last, time_elapsed - last_time_elapsed)
        knee_pos_last = knee_data_1["MulPOS"]
        jump_duration_data = {"jump_duration": jump_duration}

        csvData_shoulder.append({**shoulder_data_2, **shoulder_data_1, **shoulder_INPtorque, **shoulder_goalpos, **fps_data, **jump_duration_data})
        csvData_knee.append({**knee_data_2, **knee_data_1, **knee_INPtorque, **knee_goalpos})
        
        print("step:{}\t{}℃  SHDR: {:.0f}-->{:.0f}\t torque: {:.0f}\t {}℃  KNEE: {:.0f}-->{:.0f}\t torque: {:.0f}\t JUMPS: {} Duration:{:.2f}\t\t TIME ELAPSED: {:.2f}\t FPS: {:.0f}\t PDI: {:.2f},{:.2f},{:.2f}".format(i, shoulder_data_2["TMP"], shoulder_data_1["MulPOS"], shoulder_pos, shoulder_torque, knee_data_2["TMP"], knee_data_1["MulPOS"], knee_pos, knee_torque, jump_count, jump_duration, t, fps, p_gains, d_gains, i_gains))
        
        if abs(knee_data_1["MulPOS"] - knee_pos) < 2 and abs(shoulder_data_1["MulPOS"] - shoulder_pos) < 2:
            if knee_pos == Solo_Init[0] and shoulder_pos == Solo_Init[1]:
                print("\n\n JUMP")
                knee_pos = Solo_Jump[0]
                shoulder_pos = Solo_Jump[1]
                p_gains = 5
                d_gains = 1
                i_gains = 5
                """
                p_gains = jump_pdi[0]
                d_gains = jump_pdi[1]
                i_gains = jump_pdi[2]
                """

            elif knee_pos == Solo_Jump[0] and shoulder_pos == Solo_Jump[1]:
                print("\n\n INIT--Its Going Down")
                knee_pos = Solo_Init[0]
                shoulder_pos = Solo_Init[1]
                p_gains = 1
                d_gains = 1
                i_gains = 5
                """
                p_gains = init_pdi[0]
                d_gains = init_pdi[1]
                i_gains = init_pdi[2]
                """
                jump_count += 1
                last_time_elapsed_jump = time_elapsed_jump
                time_elapsed_jump = time.time()

        init_pdi, jump_pdi = PDI_Gain_update()

        if safety(knee_data_1["MulPOS"], shoulder_data_1["MulPOS"]) == 1:
            break
    
    initialize(motorNum=motorNum)

    #print("SHOULDER MAXSPEED: ",max(csvData_shoulder, key=lambda x: x['SPD'])['SPD'])
    #print("KNEE     MAXSPEED: ",max(csvData_knee, key=lambda x: x['SPD'])['SPD'])
    fps = [data['fps'] for data in csvData_shoulder]
    print("FPS              : ", round(statistics.median(fps)))
    Jump = [data['jump_duration'] for data in csvData_shoulder[3:]]
    print("Jump Duration    : ", statistics.median(Jump))
    Solo_plot_graph(csvData_shoulder, csvData_knee)

def SOLO_test_policy():
    num_steps = 100000
    start_time = time.time()
    knee_pos = 0
    shoulder_pos = 0
    actuator_network_path = "/home/sj/Desktop/SHADYcodes/RMD-X8/logs/Dynamic-XL.pt"
    actuator_network = torch.jit.load(actuator_network_path).to('cpu')
    pos_t   = torch.zeros(1, 2)
    pos_tt  = torch.zeros(1, 2)
    pos_ttt = torch.zeros(1, 2)
    vel_t   = torch.zeros(1, 2)
    vel_tt  = torch.zeros(1, 2)
    vel_ttt = torch.zeros(1, 2)
    torques = torch.zeros(1, 2)
    shoulder_torque = 0
    knee_torque = 0
    jump_count = 0
    time_elapsed_jump = 0
    last_time_elapsed_jump = 0
    time_elapsed = 0

    initialize(motorNum=motorNum)

    for i in range(num_steps):
        shoulder_data_2 = shoulder_MTR.torque_closed_loop(shoulder_torque)
        knee_data_2 = knee_MTR.torque_closed_loop(knee_torque)
        shoulder_data_1 = shoulder_MTR.read_multi_turns_angle()
        knee_data_1 = knee_MTR.read_multi_turns_angle()

        shoulder_data_1["MulPOS"] = shoulder_data_1["MulPOS"]/900
        knee_data_1["MulPOS"] = knee_data_1["MulPOS"]/900

        pos_t[0][0] = shoulder_data_1["MulPOS"] - shoulder_pos
        vel_t[0][0] = shoulder_data_2["vel"]
        pos_t[0][1] = knee_data_1["MulPOS"] - knee_pos
        vel_t[0][1] = knee_data_2["vel"]



        input_data = [pos_t.unsqueeze(-1), pos_tt.unsqueeze(-1), pos_ttt.unsqueeze(-1), vel_t.unsqueeze(-1), vel_tt.unsqueeze(-1), vel_ttt.unsqueeze(-1)]
        input_data = torch.cat(input_data, dim=-1)

        torques = actuator_network(input_data)
        shoulder_torque = round(torques[0,0].item())
        knee_torque = round(torques[0,1].item())

        shoulder_INPtorque = {"INTorque": shoulder_torque}
        shoulder_goalpos = {"GoalPOS": shoulder_pos}
        shoulder_data_2.pop("pos")
        knee_INPtorque = {"INTorque": knee_torque}
        knee_goalpos = {"GoalPOS": knee_pos}
        knee_data_2.pop("pos")
        last_time_elapsed = time_elapsed
        time_elapsed = time.time()
        t =  time_elapsed - start_time
        fps = 1 / (time_elapsed - last_time_elapsed)
        fps_data = {"fps": fps}

        pos_tt    = torch.clone(pos_t)
        pos_ttt   = torch.clone(pos_tt)
        vel_tt    = torch.clone(vel_t)
        vel_ttt   = torch.clone(vel_tt)

        jump_duration = time_elapsed_jump - last_time_elapsed_jump
        jump_duration_data = {"jump_duration": jump_duration}
        
        csvData_shoulder.append({**shoulder_data_2, **shoulder_data_1, **shoulder_INPtorque, **shoulder_goalpos, **fps_data, **jump_duration_data})
        csvData_knee.append({**knee_data_2, **knee_data_1, **knee_INPtorque, **knee_goalpos})
        
        print("step:{},\tSHDR\t input angle: {:.0f},\t angle: {:.0f},\t speed: {:.0f},\t torque: {:.0f},\tJUMPS: {} Duration:{}\tTEMP: {} \t\t TIME ELAPSED: {:.2f}\t FPS: {:.0f}".format(i, shoulder_pos, shoulder_data_1["MulPOS"], shoulder_data_2["vel"], shoulder_torque, jump_count, jump_duration, shoulder_data_2["TMP"], t, fps))
        print("\t\tKNEE\t input angle: {:.0f},\t angle: {:.0f},\t speed: {:.0f},\t torque: {:.0f}\t\t\t\t\t\tTEMP: {}".format(knee_pos, knee_data_1["MulPOS"], knee_data_2["vel"], knee_torque, knee_data_2["TMP"]))
    
        if abs(knee_data_1["MulPOS"] - knee_pos) < 2 and abs(shoulder_data_1["MulPOS"] - shoulder_pos) < 2:
            shoulder_pos = random.randint(SHOULDER_RANGE[0] + SAFETY_OFFSET + 2, SHOULDER_RANGE[1] - SAFETY_OFFSET - 2)
            knee_pos = random.randint(KNEE_RANGE[0] + SAFETY_OFFSET + 2, KNEE_RANGE[1] - SAFETY_OFFSET - 2)
            jump_count += 1
            last_time_elapsed_jump = time_elapsed_jump
            time_elapsed_jump = time.time()

        if safety(knee_data_1["MulPOS"], shoulder_data_1["MulPOS"]) == 1:
            break
    
    initialize(motorNum=motorNum)

    print("SHOULDER MAXSPEED: ",max(csvData_shoulder, key=lambda x: x['SPD'])['SPD'])
    print("KNEE     MAXSPEED: ",max(csvData_knee, key=lambda x: x['SPD'])['SPD'])
    fps = [data['fps'] for data in csvData_shoulder]
    print("FPS              : ", round(statistics.median(fps)))
    Jump = [data['jump_duration'] for data in csvData_shoulder[3:]]
    print("Jump Duration    : ", statistics.median(Jump))
    
    df_sh = pd.DataFrame.from_dict(csvData_shoulder) 
    df_kn = pd.DataFrame.from_dict(csvData_knee) 
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].set_title('Shoulder Data')
    axs[1].set_title('Knee Data')
    axs[0].legend()
    axs[1].legend()
    df_sh.plot(y=['MulPOS', 'GoalPOS'], ax=axs[0])
    df_kn.plot(y=['MulPOS', 'GoalPOS'], ax=axs[1])

    plt.tight_layout(pad=3.0)
    plt.show()

def Solo_plot_graph(csvData_shoulder, csvData_knee):
    df_sh = pd.DataFrame.from_dict(csvData_shoulder) 
    df_kn = pd.DataFrame.from_dict(csvData_knee)
    headers_shoulder = df_sh[['TRQ', 'SPD', 'MulPOS', 'INTorque', 'GoalPOS']]
    headers_knee = df_kn[['TRQ', 'SPD', 'MulPOS', 'INTorque', 'GoalPOS']]
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].plot(headers_shoulder)
    axs[0].set_title('Shoulder Data')
    axs[0].legend(headers_shoulder.columns)  # add legend with column names
    axs[1].plot(headers_knee)
    axs[1].set_title('Knee Data')
    axs[1].legend(headers_knee.columns)  # add legend with column names
    plt.tight_layout(pad=3.0)
    plt.show()


    new_headers_shoulder = ['TMP_1', 'TRQ_1', 'SPD_1', 'MulPOS_1', 'INTorque_1', 'GoalPOS_1', 'FramesPerSecond', 'JumpDuration', 'Height']
    shoulder_data = []
    for d in csvData_shoulder:
        new_dict = dict(zip(new_headers_shoulder, d.values()))
        shoulder_data.append(new_dict)

    new_headers_knee = ['TMP_2', 'TRQ_2', 'SPD_2', 'MulPOS_2', 'INTorque_2', 'GoalPOS_2']
    knee_data = []
    for d in csvData_knee:
        new_dict = dict(zip(new_headers_knee, d.values()))
        knee_data.append(new_dict)

    data_sea = []
    for i in range(len(shoulder_data)):
        # Create a new dictionary by merging the two dictionaries
        new_dict = {**shoulder_data[i], **knee_data[i]}
        data_sea.append(new_dict)
    
    collected_data = pd.DataFrame.from_dict(data_sea) 

    answer = input("\n\nWant to save this data?(y/n):")
    if answer.lower() != "n":
        filename = input("Enter file name: ")
        collected_data.to_csv (r'./logs/'+ filename +'.csv', mode='a', index = False, header=True)
        print("Data Succesfuly saved to ",filename,".csv")
    else:
        print("\nDATA DISCARDED",answer)

def data_collection_using_PID_control():
    num_steps = 100000
    start_time = time.time()
    shoulder_pos = Solo_Init[1]
    shoulder_data_1 = {"MulPOS": 0}
    shoulder_data_2 = {"vel": 0}
    p_gains = 8
    d_gains = 0.1
    i_gains = 0.1
    shoulder_last_error = 0
    shoulder_integral = 0

    initialize(motorNum=motorNum)
    
    for i in range(num_steps):
        time_elapsed = time.time()
        t =  time_elapsed - start_time

        shoulder_data_1 = shoulder_MTR.read_multi_turns_angle()
        shoulder_data_1["MulPOS"] = shoulder_data_1["MulPOS"]/900

        shoulder_error = shoulder_pos - shoulder_data_1["MulPOS"]
        shoulder_integral += shoulder_error * DT
        shoulder_derivative = (shoulder_error - shoulder_last_error) / DT
        shoulder_last_error = shoulder_error
        shoulder_torque = p_gains * shoulder_error + i_gains * shoulder_integral + d_gains * shoulder_derivative

        shoulder_torque = round(min(max(shoulder_torque, TORQUE_RANGE[0]), TORQUE_RANGE[1]))
        shoulder_data_2 = shoulder_MTR.torque_closed_loop(round(shoulder_torque))

        shoulder_INPtorque = {"INTorque": shoulder_torque}
        shoulder_goalpos = {"GoalPOS": shoulder_pos}
        shoulder_data_2.pop("pos")
        
        csvData_shoulder.append({**shoulder_data_2, **shoulder_data_1, **shoulder_INPtorque, **shoulder_goalpos})

        print("step:{},\tSHDR\t input angle: {:.0f},\t angle: {:.0f},\t speed: {:.0f},\t torque: {:.0f},\t Time Step: {:.0f},\t Time Elapsed: {:.0f}".format(i, shoulder_pos, shoulder_data_1["MulPOS"], shoulder_data_2["vel"], shoulder_torque, t, time.time() - start_time))

        if abs(shoulder_data_1["MulPOS"] - shoulder_pos) < 2:
            if shoulder_pos == Solo_Init[1]:
                print("\n\n JUMP")  
                shoulder_pos = Solo_Jump[1]
                p_gains = 10
                d_gains = 0.05
                i_gains = 0.1


            elif shoulder_pos == Solo_Jump[1]:
                print("\n\n INIT")
                shoulder_pos = Solo_Init[1]
                p_gains = 2
                d_gains = 0.1
                i_gains = 50

        if safety(0, shoulder_data_1["MulPOS"]) == 1:
            break

        time.sleep(DT)

    df = pd.DataFrame.from_dict(csvData_shoulder) 
    df.to_csv (r'./test_T1.csv', index = False, header=True)
    df.plot()
    plt.show()

def test_MPD_control():
    num_steps = 1000000
    shoulder_pos = Solo_Init[1]
    shoulder_data_1 = {"MulPOS": 0}
    shoulder_data_2 = {"vel": 0}
    shoulder_last_error = 0
    shoulder_integral = 0    
    p_gains = 25
    d_gains = 0.8
    i_gains = 15
    f_imp = 0.0
    time_elapsed = 0
    shoulder_torque = 0
    for i in range(num_steps):
        shoulder_data_2 = Test_MTR.torque_closed_loop(shoulder_torque)
        shoulder_data_1 = Test_MTR.read_multi_turns_angle()
        shoulder_data_1["MulPOS"] = shoulder_data_1["MulPOS"]/900

        shoulder_error = shoulder_pos - shoulder_data_1["MulPOS"]
        shoulder_integral += shoulder_error * DT
        shoulder_derivative = (shoulder_error - shoulder_last_error) / DT
        shoulder_last_error = shoulder_error
        shoulder_torque = p_gains * shoulder_error + i_gains * shoulder_integral + d_gains * shoulder_derivative
        shoulder_torque = round(min(max(shoulder_torque, TORQUE_RANGE[0]), TORQUE_RANGE[1]))

        shoulder_INPtorque = {"INTorque": shoulder_torque}
        shoulder_goalpos = {"GoalPOS": shoulder_pos}
        shoulder_data_2.pop("pos")
        
        csvData_shoulder.append({**shoulder_data_2, **shoulder_data_1, **shoulder_INPtorque, **shoulder_goalpos})
        last_time_elapsed = time_elapsed
        time_elapsed = time.time()
        fps = FPS_calc(time_elapsed, last_time_elapsed)
        print("step:{},\tTEST\t input angle: {:.0f},\t angle: {:.0f},\t speed: {:.0f},\t torque: {:.0f},\t Time Elapsed: {:.0f},\t FPS: {:.0f}".format(i, shoulder_pos, shoulder_data_1["MulPOS"], shoulder_data_2["vel"], f_imp, time_elapsed, fps))

        if abs(shoulder_data_1["MulPOS"] - shoulder_pos) < 2:
            if shoulder_pos == Solo_Init[1]:
                print("\n\n JUMP")  
                shoulder_pos = Solo_Jump[1]

            elif shoulder_pos == Solo_Jump[1]:
                print("\n\n INIT")
                shoulder_pos = Solo_Init[1]

        if safety(0, shoulder_data_1["MulPOS"]) == 1:
            break

    initialize(motorNum=motorNum)
    plot_graph(csvData_shoulder)

def data_collection_using_other_MNN_control():
    num_steps = 10000
    start_time = time.time()
    k = 0
    pos = 0
    angles = [0, 20, 90, 50, 180, 30, 60, 10, 200]

    actuator_network_path = "/home/sj/Desktop/SHADYcodes/ROBODOG/resources/actuator_nets/unitree_go1.pt"
    actuator_network = torch.jit.load(actuator_network_path).to('cpu')
    pos_t   = torch.zeros(1, 1)
    pos_tt  = torch.zeros(1, 1)
    pos_ttt = torch.zeros(1, 1)
    vel_t   = torch.zeros(1, 1)
    vel_tt  = torch.zeros(1, 1)
    vel_ttt = torch.zeros(1, 1)
        
    for i in range(num_steps):
        time_elapsed = time.time()
        t =  time_elapsed - start_time

        posdata = shoulder_MTR.read_multi_turns_angle()
        posdata["MulPOS"] = round(posdata["MulPOS"]/900)
        if posdata["MulPOS"] == 80063993375475:
            posdata["MulPOS"] = 8
        if i > 3:
            pos_t[0][:]   = posdata["MulPOS"] - pos
            vel_t[0][:]   = data["vel"]

        input_data = [pos_t, pos_tt, pos_ttt, vel_t, vel_tt, vel_ttt]
        input_data = torch.cat(input_data, dim=1)
        torques = actuator_network(input_data)
        data = shoulder_MTR.torque_closed_loop(round(torques[0][0].item()))

        try :
            if (pos == posdata["MulPOS"]):
                if k >= len(angles):
                    k = 0
                pos = angles[k]
                k += 1
        except:
            pass
        
        INPtorque = {"INTorque": round(torques[0][0].item())}
        goalpos = {"GoalPOS": pos}
        data.pop("pos")
        csvData.append({**data, **posdata, **INPtorque, **goalpos})

        pos_tt    = torch.clone(pos_t)
        pos_ttt   = torch.clone(pos_tt)
        vel_tt    = torch.clone(vel_t)
        vel_ttt   = torch.clone(vel_tt)

        print("step:{}, angle: {:.0f}, speed: {:.0f}, torque: {:.0f} Time Step: {:.0f}, Time Elapsed: {:.0f}".format(i, posdata["MulPOS"], vel_t, torques, t, time.time() - start_time))
        time.sleep(DT)

    df = pd.DataFrame.from_dict(csvData) 
    df.to_csv (r'./logs/actuators/DataCollection/Experiment_NO.1/test_T1.csv', index = False, header=True)
    df.plot()
    plt.show()

def data_collection_using_position_control():
    num_steps = 5000
    start_time = time.time()
    j = 1
    k = 0
    speed = 1000
    pos = 0
    angles = [0, 90, 360]
    for i in range(num_steps):
        time_elapsed = time.time()
        t =  time_elapsed - start_time
        freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * t / (DT * num_steps)
        ampl = MAX_AMPL + (MIN_AMPL - MAX_AMPL) * t / (DT * num_steps)
        speed = abs(round(ampl * math.sin(2 * math.pi * freq * t)))

        data = shoulder_MTR.position_closed_loop_2(speed, math.floor(pos))
        posdata = shoulder_MTR.read_multi_turns_angle()
        posdata["MulPOS"] = round(posdata["MulPOS"]/900)
        if posdata["MulPOS"] == 80063993375475:
            posdata["MulPOS"] = 8
        
        torque = data["trq"]
        goalpos = {"GoalPOS": pos}
        data.pop("pos")
        csvData.append({**data, **posdata, **goalpos})

        print("step:{} input angle: {:.0f}, angle: {:.0f}, speed: {:.0f}, torque: {:.0f} Time Step: {:.0f}, Time Elapsed: {:.0f}".format(i, pos, posdata["MulPOS"], speed, torque, t, time.time() - start_time))

        try :
            if (pos == posdata["MulPOS"]):
                if k >= len(angles):
                    k = 0
                pos = angles[k]
                k += 1
        except:
            print("input angle not changed")

        # Sleep for the time step
        time.sleep(DT)
        j += 10 * k

    df = pd.DataFrame.from_dict(csvData) 
    df.to_csv (r'./logs/actuators/DataCollection/Experiment_NO.1/test_T1.csv', index = False, header=True)
    df.plot()
    plt.show()

def data_collection_using_torque_control():
    MAX_FREQ = 1
    MIN_FREQ = 0.1
    MAX_AMPL = 40
    MIN_AMPL = -40

    num_steps = 5000
    start_time = time.time()
    j = 1
    k = 0
    speed = 1000
    for i in range(num_steps):
        time_elapsed = time.time()
        t =  time_elapsed - start_time
        freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * t / (DT * num_steps)
        ampl = MAX_AMPL + (MIN_AMPL - MAX_AMPL) * t / (DT * num_steps)
        torque = round(ampl * math.sin(2 * math.pi * freq * t))

        data = shoulder_MTR.torque_closed_loop(torque)
        posdata = shoulder_MTR.read_multi_turns_angle()
        posdata["MulPOS"] = round(posdata["MulPOS"]/900)
        if posdata["MulPOS"] == 80063993375475:
            posdata["MulPOS"] = 8
        
        INPtorque = {"INTorque": torque}
        data.pop("pos")
        
        csvData.append({**data, **posdata, **INPtorque})

        print("step:{}, angle: {:.0f}, speed: {:.0f}, torque: {:.0f} Time Step: {:.0f}, Time Elapsed: {:.0f}".format(i, posdata["MulPOS"], speed, torque, t, time.time() - start_time))

        # Sleep for the time step
        time.sleep(DT)
        j += 10 * k

    df = pd.DataFrame.from_dict(csvData) 
    df.to_csv (r'./logs/actuators/DataCollection/Experiment_NO.1/test_T1.csv', index = False, header=True)
    df.plot()
    plt.show()

def test_policy():
    num_steps = 50000
    start_time = time.time()
    k = 0
    pos = 0
    angles = [-85, 55, 10, 50, -30, 20, -10, 30]

    actuator_network_path = "/home/sj/Desktop/SHADYcodes/ROBODOG/resources/actuator_nets/RMDx8_V1.pt"
    actuator_network = torch.jit.load(actuator_network_path).to('cpu')
    pos_t   = torch.zeros(1, 1)
    pos_tt  = torch.zeros(1, 1)
    pos_ttt = torch.zeros(1, 1)
    vel_t   = torch.zeros(1, 1)
    vel_tt  = torch.zeros(1, 1)
    vel_ttt = torch.zeros(1, 1)
    pos_goal= torch.zeros(1, 1)
    
    initialize(motorNum=motorNum)

    for i in range(num_steps):
        time_elapsed = time.time()
        t =  time_elapsed - start_time

        posdata = Test_MTR.read_multi_turns_angle()
        
        posdata["MulPOS"] = round(posdata["MulPOS"]/900)

        if i > 3:
            pos_t[0][:]   = posdata["MulPOS"] - pos
            vel_t[0][:]   = data["vel"]

        input_data = [pos_t, pos_tt, pos_ttt, vel_t, vel_tt, vel_ttt]
        input_data = torch.cat(input_data, dim=1)

        torques = actuator_network(input_data)
        data = Test_MTR.torque_closed_loop(round(torques[0][0].item()))
        

        try :
            if abs(posdata["MulPOS"] - pos) < 2:
                if k >= len(angles):
                    k = 0
                pos = angles[k]
                k += 1
        except:
            pass


        goalpos = {"GoalPOS": pos}
        csvData.append({**data, **posdata, **goalpos})

        pos_tt    = torch.clone(pos_t)
        pos_ttt   = torch.clone(pos_tt)
        vel_tt    = torch.clone(vel_t)
        vel_ttt   = torch.clone(vel_tt)

        #print("{:.0f}, {:.0f}".format(pos, data["trq"]))
        print("inangle: {}, angle: {}".format(pos, posdata["MulPOS"]))
        if safety(0, posdata["MulPOS"]):
            break
    
    plot_graph(csvData)

def Solo_easy_control():
    initialize(motorNum=motorNum)
    i = 0
    time_elapsed = 0

    while 1:
        shoulder_MTR.torque_closed_loop(0)
        knee_MTR.torque_closed_loop(0)
        shoulder_pos = shoulder_MTR.read_multi_turns_angle()
        knee_pos = knee_MTR.read_multi_turns_angle()

        last_time_elapsed = time_elapsed
        time_elapsed = time.time()
        fps = FPS_calc(time_elapsed, last_time_elapsed)

        try:
            print("time: {:.0f}\t JOINT: Shoulder\t angle: {}\t JOINT: Knee\t angle: {},\t fps: {}".format(i, shoulder_pos["MulPOS"]/900, knee_pos["MulPOS"]/900,fps))
        except:
            print("fps: ",fps)
        i += 1
        
        if safety(knee_pos["MulPOS"]/900, shoulder_pos["MulPOS"]/900) == 1:
            break
            
    print("\n-------EasyControl Exit-------\n")

def easy_control():
    initialize(motorNum=motorNum)
    i = 0
    while 1:
        shoulder_MTR.torque_closed_loop(0)
        shoulder_pos = shoulder_MTR.read_multi_turns_angle()
        print("time: {:.0f}\t JOINT: Shoulder\t angle: {}\t".format(i, shoulder_pos["MulPOS"]/900))
        i += 1
        
        if safety(0, shoulder_pos["MulPOS"]/900) == 1:
            break


    print("\n-------EasyControl Exit-------\n")

def easy_control_pos():
    i = 0
    while 1:
        shoulder_MTR.position_closed_loop_2(1000,-11)
        shoulder_pos = shoulder_MTR.read_multi_turns_angle()
        knee_MTR.position_closed_loop_2(2000,-3)
        knee_pos = knee_MTR.read_multi_turns_angle()


        print("time: {:.0f}\t JOINT: Shoulder\t angle: {}".format(i, shoulder_pos["MulPOS"]/900))
        print("time: {:.0f}\t JOINT: Knee    \t angle: {}".format(i, knee_pos["MulPOS"]/900))
        
        time.sleep(0.2)

        shoulder_MTR.position_closed_loop_2(1000,-50)
        shoulder_pos = shoulder_MTR.read_multi_turns_angle()
        knee_MTR.position_closed_loop_2(2000,66)
        knee_pos = knee_MTR.read_multi_turns_angle() 


        print("time: {:.0f}\t JOINT: Shoulder\t angle: {}".format(i, shoulder_pos["MulPOS"]/900))
        print("time: {:.0f}\t JOINT: Knee    \t angle: {}".format(i, knee_pos["MulPOS"]/900))

        time.sleep(0.25)
        
        #if keyboard.is_pressed('escape'):  # if key 'escape' is pressed 
        #    initialize()
        #    time.sleep(DT)
        #    break

        i += 1

def initialize(motorNum):
    if initializing == True:
        print("------- INITIALIZING -------\n")
        while(True):
            shoulder_offset = -11
            knee_offset = 10

            #if motorNum == 3:
            #    Test_MTR.position_closed_loop_2(1000, shoulder_offset)
            #    test_pos = Test_MTR.read_multi_turns_angle()

               # print("\rJOINT: Test\t angle: {}".format(test_pos["MulPOS"]/900),end = ' ')
               # if (abs(test_pos["MulPOS"]/900 - shoulder_offset) < 1):
               #S     break

            if motorNum == 2:
                shoulder_MTR.position_closed_loop_2(1000, shoulder_offset)
                knee_MTR.position_closed_loop_2(1000, knee_offset)
                shoulder_posdata = shoulder_MTR.read_multi_turns_angle()
                knee_posdata = knee_MTR.read_multi_turns_angle()

                print("\rJOINT: Shoulder\t angle: {}\t JOINT: Knee\t angle: {}".format(shoulder_posdata["MulPOS"]/900, knee_posdata["MulPOS"]/900),end = ' ')
                if (abs(shoulder_posdata["MulPOS"]/900 - shoulder_offset) < 1 and abs(knee_posdata["MulPOS"]/900 - knee_offset) < 1):
                    break

            elif motorNum == 1:
                shoulder_MTR.position_closed_loop_2(1000, shoulder_offset)
                shoulder_posdata = shoulder_MTR.read_multi_turns_angle()

                print("\rJOINT: Shoulder\t angle: {}\t JOINT: Knee \t angle: {}".format(shoulder_posdata["MulPOS"]/900, knee_posdata["MulPOS"]/900),end = ' ')
                if (abs(shoulder_posdata["MulPOS"]/900 - shoulder_offset) < 1):
                    break
                
def safety(knee_pos, shoulder_pos):
    if keyboard.is_pressed('escape'):
        initialize(motorNum=motorNum)
        return 1

    if safety_activate == False:
        return 0

    if knee_pos < KNEE_RANGE[0] + SAFETY_OFFSET or knee_pos > KNEE_RANGE[1] - SAFETY_OFFSET:
        print("\n ERROR: KNEE motor position in danger area:", knee_pos,"\tMaximum point: ",KNEE_RANGE[0],"~",KNEE_RANGE[1])
        initialize(motorNum=motorNum)
        return 1

    if shoulder_pos < SHOULDER_RANGE[0] + SAFETY_OFFSET or shoulder_pos > SHOULDER_RANGE[1] - SAFETY_OFFSET:
        print("\n ERROR: SHOULDER motor position in danger area:", shoulder_pos,"\tMaximum point: ",SHOULDER_RANGE[0],"~",SHOULDER_RANGE[1])
        initialize(motorNum=motorNum)
        return 1
    
def plot_graph(csvData):
    df = pd.DataFrame.from_dict(csvData) 
    df.to_csv (r'./logs/stest_T1.csv', index = False, header=True)
    df.plot()
    plt.show()

    answer = input("\n\nWant to save this data?(y/n):")
    if answer.lower() == 'y':
            df.to_csv (r'./logs/test_MTR.csv', mode='a', index = False, header=True)
            print("Data Succesfuly saved")
    else:
        print("\nDATA DISCARDED",answer)

def FPS_calc(time_elapsed, last_time_elapsed):
    fps = 1 / (time_elapsed - last_time_elapsed)
    return fps

def _get_obs():
    """
    RETURNS PHYSICAL OBS FOR B-NN
    -----------------------------

    RETURN [SH_POS, NE_POS, SH_VEL, KNEE_VEL]
    """
    shoulder_data_2 = shoulder_MTR.read_motor_status_2()
    knee_data_2 = knee_MTR.read_motor_status_2()
    shoulder_data_1 = shoulder_MTR.read_multi_turns_angle()
    knee_data_1 = knee_MTR.read_multi_turns_angle()

    shoulder_data_1["MulPOS"] = shoulder_data_1["MulPOS"]/900
    knee_data_1["MulPOS"] = knee_data_1["MulPOS"]/900

    input_data = [shoulder_data_1["MulPOS"], knee_data_1["MulPOS"], shoulder_data_2["vel"], knee_data_2["vel"]]

    obs = input_data

    if safety(knee_data_1["MulPOS"], shoulder_data_1["MulPOS"]) == 1:
        return 1

    return obs

def shoulder_torque_control(torque):
    shoulder_MTR.torque_closed_loop(0.1 * round(torque))

def knee_torque_control(torque):
    knee_MTR.torque_closed_loop(0.1 * round(torque))

def cal_velocity(pos, lastpos, timelapsed):
    return ((pos - lastpos) * math.pi / 180) / timelapsed

initializing = True
motorNum=2
safety_activate = True

try:
    #shoulder_MTR.bus.flush_tx_buffer()
    #knee_MTR.bus.flush_tx_buffer()
    #initialize(motorNum=motorNum)
    #Solo_data_collection_using_kinematics_PID_control()
    Solo_easy_control()
    #Solo_data_collection_using_PID_control()
    #SOLO_test_policy()
    #test_MPD_control()
    #test_policy()
    #test_FPS_calc()
    #data_collection_using_PID_control()
    #print(_get_obs())

except Exception as e: 
    print("\n-------CODE ERROR: RETURNING TO BASE-------\n")
    initialize(motorNum=motorNum)
    print(traceback.format_exc())