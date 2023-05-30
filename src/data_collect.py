from rmd_x8 import RMD_X8
import time
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import math
import numpy as np
import traceback
import statistics
from kinamatics import SOLO_kinematics
import threading
import scipy.signal as signal
from matplotlib.animation import FuncAnimation
import keyboard
import json
import csv
import threading

class SOLO_Controller:
    def __init__(self) -> None:
        self.IK = SOLO_kinematics() 
        _ = threading.Thread(target=self.IK.render, name='Animation')
        _.start()

        self.shoulder_MTR = RMD_X8(0x0141)
        self.shoulder_MTR.setup("0")

        self.knee_MTR = RMD_X8(0x0142)
        self.knee_MTR.setup("0")

        self.initialize_bool = True
        self.init_pose = np.array([-11, 10])
        
        self.TORQUE_RANGE = [-300,300]
        self.KNEE_RANGE = [-19,90]
        self.SHOULDER_RANGE = [-90,60]
        self.SAFETY_OFFSET = 4
        self.MOTOR_SHIFT = [0, 0]
        self.cycle_num = np.array([-1, 0])
        
        self.NUM_STEPS = 100000
        self.HZ = 255                       # actual running on 200HZ
        self.DT = 1/self.HZ
        self.freq_start = 0.1
        self.freq_inc = (3 - 0.1) / self.NUM_STEPS
        self.elapsed_time = 0

        self.dof_num = 2
        self.dof_pos = np.zeros(self.dof_num)
        self.dof_vel = np.zeros(self.dof_num)
        self.dof_pos_last = np.zeros(self.dof_num)
        self.dof_trq_command = np.zeros(self.dof_num)
        self.dof_enc_pos = np.zeros(self.dof_num)
        self.dof_enc_pos_last = np.zeros(self.dof_num)
        self.torque_PID = np.zeros(self.dof_num)
        self.dof_pos_target = np.zeros(self.dof_num)

        self.integral = np.zeros(self.dof_num)
        self.derivative = np.zeros(self.dof_num)
        self.last_error = np.zeros(self.dof_num)
        
        self.p_gain = self.IK.customs[0][1] = 2.4
        self.i_gain = self.IK.customs[0][1] = 2.0
        self.d_gain = self.IK.customs[0][1] = 0.13

        self.sh_p_gain = self.IK.customs[0][0] = 2.4
        self.sh_i_gain = self.IK.customs[0][1] = 2.0
        self.sh_d_gain = self.IK.customs[0][2] = 0.13
        self.kn_p_gain = self.IK.customs[1][0] = 2.4
        self.kn_i_gain = self.IK.customs[1][1] = 2.0
        self.kn_d_gain = self.IK.customs[1][2] = 0.13

        # 4, 4, 0.4 bigboyjump_1000HZ -- fs100, cutoff2, nyquist0.8, order3
        # 7, 7, 0.4 SweepStand_300HZ  --  fs20, cutoff2, nyquist0.8, order3

        fs = [20, 20]                                                                    # sampling frequency
        cutoff_freq = [2, 2]                                                             # cutoff frequency in Hz
        order = [1, 2]                                                                   # filter order
        nyquist_freq = [0.8 * fs[0], 0.8 * fs[1]]
        cutoff_norm = [cutoff_freq[0] / nyquist_freq[0], cutoff_freq[1] / nyquist_freq[1]]                                    # normalized cutoff frequency
        
        self.a = np.zeros(self.dof_num)
        self.b = np.zeros(self.dof_num)
        self.b0, self.a0 = signal.butter(order[0], cutoff_norm[0], btype='lowpass')
        self.b1, self.a1 = signal.butter(order[1], cutoff_norm[1], btype='lowpass')
        
        self.torque_input_filter_buf = np.zeros([self.dof_num, 1000])
        self.torque_ouput_filter_buf = np.zeros([self.dof_num, 1000])

        data_list = [{'TMP': 0, 'TRQ': 0, 'SPD': 0, 'POS': 0}]
        self.data = np.zeros([self.NUM_STEPS, 2], dtype=object)
        self.data[:, :] = data_list

        with open("/home/sj/Desktop/SHADYcodes/RMD-X8/Motors_Dict.json", "r") as f:
            command_to_torque = json.load(f)
        self.torque_constant = [command_to_torque["RMD-X8-PRO-SHOULDER"]["torque constant"] , command_to_torque["RMD-X8-V2-KNEE"]["torque constant"]]

    def apply_lowpass_filter(self):
        pos_error = self.dof_pos_target - self.dof_pos
        self.integral += pos_error * self.DT
        self.derivative = (pos_error - self.last_error) / self.DT
        self.last_error = pos_error
        self.torque_PID = self.p_gain * pos_error + self.i_gain * self.integral + self.d_gain * self.derivative

        self.torque_input_filter_buf[:, :-1] = self.torque_input_filter_buf[:, 1:]
        self.torque_input_filter_buf[:, -1] = self.torque_PID[:]

        # Apply the filter to the input buffer
        self.torque_ouput_filter_buf[:, :-1] = self.torque_ouput_filter_buf[:, 1:]
        self.torque_ouput_filter_buf[0, -1] = signal.lfilter(self.b0, self.a0, self.torque_input_filter_buf)[0, -1]
        self.torque_ouput_filter_buf[1, -1] = signal.lfilter(self.b1, self.a1, self.torque_input_filter_buf)[1, -1]
        self.dof_trq_command = np.round(self.torque_ouput_filter_buf[:, -1])
        self.dof_trq_command = np.clip(self.dof_trq_command, self.TORQUE_RANGE[0], self.TORQUE_RANGE[1])
        # self.dof_trq_command = np.round(self.torque_PID)
        # self.dof_trq_command = np.clip(self.torque_PID, self.TORQUE_RANGE[0], self.TORQUE_RANGE[1])

    def Solo_data_collection_using_kinematics_PID_control(self):
        time_elapsed = 0
        self.initialize()

        print("Press Enter to start PID control...")
        while(1):
            print(self.IK.start)
            if self.IK.start == True:
                break
               
        for i in range(self.NUM_STEPS):
            self.data[i, 0] = self.shoulder_MTR.torque_closed_loop(round(self.dof_trq_command[0]))
            self.data[i, 1] = self.knee_MTR.torque_closed_loop(round(self.dof_trq_command[1]))  
            # self.data[i, 0] = self.shoulder_MTR.torque_closed_loop(0)
            # self.data[i, 1] = self.knee_MTR.torque_closed_loop(0)  

            self.dof_enc_pos[0] = self.data[i][0]["POS"] * 360 / 65535
            self.dof_enc_pos[1] = self.data[i][1]["POS"] * 360 / 65535

            self.update_motor_pos()
            self.update_motor_vel()   
            self.apply_lowpass_filter()

            self.dof_pos_target = self.IK.theta1_offset, self.IK.theta2_offset
            self.IK.theta3, self.IK.theta4 = (self.dof_pos[0] - 1) * math.pi / 180, (self.dof_pos[1] - 150) * math.pi / 180

            last_time_elapsed = time_elapsed
            time_elapsed = time.time()
            self.elapsed_time =  time_elapsed - last_time_elapsed
            fps = 1 / self.elapsed_time
            
            self.freq_start += self.freq_inc
            self.IK.frequency = self.freq_start

            self.data[i, 0]['TRQ'] /= self.torque_constant[0]
            self.data[i, 1]['TRQ'] /= self.torque_constant[1]

            dict_0 = {
                "INTorque": self.dof_trq_command[0] / self.torque_constant[0],
                "MulPOS": self.dof_pos[0] * math.pi / 180,
                "vel": self.dof_vel[0],
                "GoalPOS": self.dof_pos_target[0] * math.pi / 180
            }

            dict_1 = {
                "INTorque": self.dof_trq_command[1] / self.torque_constant[1],
                "MulPOS": self.dof_pos[1] * math.pi / 180,
                "vel": self.dof_vel[1],
                "GoalPOS": self.dof_pos_target[1] * math.pi / 180
            }

            self.data[i][0].update(dict_0)
            self.data[i][1].update(dict_1)

            self.sh_p_gain = self.IK.customs[0][0]
            self.sh_i_gain = self.IK.customs[0][1]
            self.sh_d_gain = self.IK.customs[0][2]
            self.kn_p_gain = self.IK.customs[1][0]
            self.kn_i_gain = self.IK.customs[1][1]
            self.kn_d_gain = self.IK.customs[1][2]

            print("step:{}\t {}℃  SHDR: {:.0f}°-->{:.0f}°\t torque: {:.0f}\t {}℃  KNEE: {:.0f}°-->{:.0f}°\t torque: {:.0f}\t\t FPS: {:.2f} P|{:.2f}  I|{:.2f}  D|{:.2f} // P|{:.2f}  I|{:.2f}  D|{:.2f}".format(i, self.data[i][0]["TMP"], self.dof_pos[0], self.dof_pos_target[0], self.dof_trq_command[0], self.data[i][1]["TMP"], self.dof_pos[1], self.dof_pos_target[1], self.dof_trq_command[1], fps, self.IK.customs[0][0],self.IK.customs[0][1],self.IK.customs[0][2],self.IK.customs[1][0],self.IK.customs[1][1],self.IK.customs[1][2]))

            if self.IK.exit == True:
                break
            if self.safety() == 1 or self.IK.exit == True:
                break

            time.sleep(self.DT)

        print(self.IK.customs)
        self.initialize()
        self.plot_graph(i)

    def plot_graph(self, steps):
        # {'TMP': -55, 'TRQ': 17, 'SPD': 0, 'POS': 47513, 'INTorque': 2.0, 'MulPOS': -0.1919848861185126, 'vel': 0.0, 'GoalPOS': -0.18438615555950572}
        
        sh = pd.DataFrame((self.data[:steps, 0]).tolist())
        ne = pd.DataFrame((self.data[:steps, 1]).tolist())
        
        sh_trq = sh.loc[:, ['INTorque', 'TRQ']]
        ne_trq = ne.loc[:, ['INTorque', 'TRQ']]
        sh_pos = sh.loc[:, ['MulPOS', 'GoalPOS', 'vel']]
        ne_pos = ne.loc[:, ['MulPOS', 'GoalPOS', 'vel']]

        sh.pop("POS")
        ne.pop("POS")
        sh.pop("SPD")
        ne.pop("SPD")

        
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        axs[0, 0].plot(sh_trq.values)
        axs[0, 0].set_title('Shoulder Torque')
        axs[0, 0].legend(sh_trq.columns)  # add legend with column names
        axs[1, 0].plot(sh_pos.values)
        axs[1, 0].set_title('Shoulder Position')
        axs[1, 0].legend(sh_pos.columns)  # add legend with column names

        axs[0, 1].plot(ne_trq.values)
        axs[0, 1].set_title('Shoulder Torque')
        axs[0, 1].legend(ne_trq.columns)  # add legend with column names
        axs[1, 1].plot(ne_pos.values)
        axs[1, 1].set_title('Shoulder Position')
        axs[1, 1].legend(ne_pos.columns)  # add legend with column names
        plt.tight_layout(pad=3.0)
        plt.show()


        sh = sh.rename(columns={'TRQ': 'TRQ_1', 'vel': 'VEL_1', 'MulPOS': 'MulPOS_1', 'INTorque': 'INTorque_1', 'GoalPOS': 'GoalPOS_1'})
        ne = ne.rename(columns={'TRQ': 'TRQ_2', 'vel': 'VEL_2', 'MulPOS': 'MulPOS_2', 'INTorque': 'INTorque_2', 'GoalPOS': 'GoalPOS_2'})

        new_dict = {**sh, **ne}        
        collected_data = pd.DataFrame.from_dict(new_dict) 

        answer = input("\n\nWant to save this data?(y/n):")
        if answer.lower() != "n":
            filename = input("Enter file name: ")
            collected_data.to_csv (r'logs/'+filename+'.csv', mode='w', index = False, header=True)
            print("Data Succesfuly saved to ",filename,".csv")
        else:
            print("\nDATA DISCARDED",answer)

    def Solo_easy_control(self):
        time_elapsed = 0
        self.IK.start = True
        while 1:
            self.dof_enc_pos[0] = self.shoulder_MTR.torque_closed_loop(0)["POS"] * 360 / 65535
            self.dof_enc_pos[1] = self.knee_MTR.torque_closed_loop(0)["POS"] * 360 / 65535
            self.update_motor_pos()

            self.IK.theta3, self.IK.theta4 = (self.dof_pos[0] - 1) * math.pi / 180, (self.dof_pos[1] - 150) * math.pi / 180

            last_time_elapsed = time_elapsed
            time_elapsed = time.time()
            self.elapsed_time =  time_elapsed - last_time_elapsed
            fps = 1 / self.elapsed_time

            print("sh_mtr: {:2f} {}\t ne_mtr: {:2f} {}\t\t{}\t fps: {:2f}".format(self.dof_pos[0], self.cycle_num[0], self.dof_pos[1], self.cycle_num[1], self.IK.frequency, fps))

            self.safety()
            time.sleep(self.DT)

        print("\n-------EasyControl Exit-------\n")

    def safety(self):
        if self.dof_pos[0] < self.SHOULDER_RANGE[0] + self.SAFETY_OFFSET or self.dof_pos[0] > self.SHOULDER_RANGE[1] - self.SAFETY_OFFSET:
            print("\n ERROR: SHOULDER motor position in danger area:", self.dof_pos[0],"\tMaximum point: ",self.SHOULDER_RANGE[0],"~",self.SHOULDER_RANGE[1])
            self.initialize()
            return 1
        
        elif self.dof_pos[1] < self.KNEE_RANGE[0] + self.SAFETY_OFFSET or self.dof_pos[1] > self.KNEE_RANGE[1] - self.SAFETY_OFFSET:
            print("\n ERROR: KNEE motor position in danger area:", self.dof_pos[1],"\tMaximum point: ",self.KNEE_RANGE[0],"~",self.KNEE_RANGE[1])
            self.initialize()
            return 1

    def update_motor_pos(self):
            cond1 = self.dof_enc_pos < 30
            cond2 = self.dof_enc_pos_last > 330
            cond3 = self.dof_enc_pos > 330
            cond4 = self.dof_enc_pos_last < 30

            self.cycle_num[cond1 & cond2] += 1
            self.cycle_num[cond4 & cond3] -= 1
                
            self.dof_enc_pos_last = self.dof_enc_pos.copy()
            self.dof_pos_last = self.dof_pos.copy()
            self.dof_pos = (self.dof_enc_pos / 9) + (self.cycle_num * 360 / 9)

    def update_motor_vel(self):
        self.dof_vel = ((self.dof_pos* math.pi / 180) - (self.dof_pos_last* math.pi / 180)) / self.elapsed_time

    def initialize(self):
        """
        Initialize the robot to the initial pose.
        You need to initialize the robot before you start counting multi turn position
        """
        if self.initialize_bool == False:
            return
        
        print("------- INITIALIZING -------\n")
        start = time.time()
        while(True):
            self.update_motor_pos()
            self.cycle_num = np.array([-1, 0])

            self.dof_enc_pos[0] = self.shoulder_MTR.position_closed_loop_2(500, self.init_pose[0])["POS"] * 360 / 65535
            self.dof_enc_pos[1] = self.knee_MTR.position_closed_loop_2(500, self.init_pose[1])["POS"] * 360 / 65535
            self.IK.theta3, self.IK.theta4 = (self.dof_pos[0] - 1) * math.pi / 180, (self.dof_pos[1] - 150) * math.pi / 180
            
            print("SHDR: {:.0f}°-->{:.0f}°\t\t KNEE: {:.0f}°-->{:.0f}°\t ".format(self.dof_pos[0], self.init_pose[0], self.dof_pos[1], self.init_pose[1]))
            print(self.cycle_num)
            
            # if (abs(self.dof_pos[0] - self.init_pose[0]) < 1 and abs(self.dof_pos[1] - self.init_pose[1]) < 1):
            #     break

            if time.time() - start > 3:
                break

    def show_file_moves(self):
        name = 'others/file'
        with open("/home/sj/Desktop/SHADYcodes/RMD-X8/original_file.csv", "r") as f:
            data = pd.read_csv(f)
        
        for i in range(len(data)):
            self.IK.theta3, self.IK.theta4 = (data["GoalPOS_1"][i] - (40 * math.pi/180)), (data["GoalPOS_2"][i] - (150 * math.pi/180))
            time.sleep(0.01)
        

class Torque_Calculation_Setup:
    def __init__(self) -> None:
        self.MTR = RMD_X8(0x0141)
        self.MTR.setup("0")
        self.MTR.running_command()

        with open("./Motors_Dict.json", "r") as f:
            self.data = json.load(f)

        self.motor_name = "RMD-X8-PRO-SHOULDER"
        
        self.SampleNum = 20
        self.commands = np.zeros(self.SampleNum)
        self.torque = np.zeros(self.SampleNum)

    def Torque_control(self):
        self.MTR.torque_closed_loop(0)
        i = 0
        while i < self.SampleNum:
            self.commands[i] = (i+1) * (2)
            self.torque[i] = self.MTR.torque_closed_loop(int(-1 * self.commands[i]))["TRQ"]             # Torque is negative for direction
            time.sleep(3)

            self.MTR.torque_closed_loop(0)
            try:
                self.torque[i] = input("Command {} \t Enter Torque: ".format(self.commands[i]))
            except:
                print('Retry')
                i-1
                continue

            i += 1
            
        self.MTR.torque_closed_loop(0)
        self.Plot_Graph()

    def Plot_Graph(self):
        torque_constant = self.commands / self.torque
        average = torque_constant[1:].sum() / len(torque_constant[1:])
        self.data[self.motor_name] = {"Command": self.commands.tolist(), "Torque": self.torque.tolist(), 'torque constant': average}


        with open("Motors_Dict.json", "w") as f:
            json.dump(self.data, f)

        print("Data is saved to file")
        print("torque_constant: ", average)

        plt.plot(self.commands, self.torque)
        plt.xlabel("Command")
        plt.ylabel("Torque")
        plt.title("Torque vs Command")
        plt.legend()
        plt.show()

    def manual(self):
        self.commands = np.array(self.data["RMD-X8-PRO-SHOULDER"]["Command"])
        self.torque = np.array(self.data["RMD-X8-PRO-SHOULDER"]["Torque"])
        torque_constants = self.commands[1:] / self.torque[1:]
        average = torque_constants.sum() / len(torque_constants)
        self.data[self.motor_name] = {"Command": self.commands, "Torque": self.torque, 'torque constant': average}

        print("Data is saved to file")
        print("torque_constant: ", average)

        plt.plot(self.commands, self.torque)
        plt.xlabel("Command")
        plt.ylabel("Torque")
        plt.title("Torque vs Command")
        plt.legend()
        plt.show()



try:
    solo = SOLO_Controller()
    # _ = threading.Thread(target=solo.update_motor_pos, name='Physical Position Update')
    # _.start()
    solo.shoulder_MTR.bus.flush_tx_buffer()
    solo.knee_MTR.bus.flush_tx_buffer()

    # solo.Solo_data_collection_using_kinematics_PID_control()
    solo.Solo_easy_control()
    # solo.initialize(False)

    # _ = Torque_Calculation_Setup()
    # _.manual()

except Exception as e:
    print("\n-------CODE ERROR: RETURNING TO BASE-------\n\n")
    # solo.initialize(False)
    print(traceback.format_exc())




"""
Command 10.0     Enter Torque: 0
Command 20.0     Enter Torque: 1.5
Command 30.0     Enter Torque: 1.72
Command 40.0     Enter Torque: 1.8
Command 50.0     Enter Torque: 2.40
Command 60.0     Enter Torque: 2.35
Command 70.0     Enter Torque: 2.85
Command 80.0     Enter Torque: 2.96
Command 90.0     Enter Torque: 3.51
Command 100.0    Enter Torque: 3.69
Command 110.0    Enter Torque: 4.04
Command 120.0    Enter Torque: 4.6
Command 130.0    Enter Torque: 5.2^CRetry
Command 130.0    Enter 
"""
