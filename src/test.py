import numpy as np
import matplotlib.pyplot as plt
from rmd_x8 import RMD_X8
import threading
import time
import can







"""
PRO is 141
"""









# 32
# 141 ~ 172
shoulder_MTR = RMD_X8(0x0141)
shoulder_MTR.setup("0")
# shoulder_MTR.running_command()
# time.sleep(1)

# knee_MTR = RMD_X8(0x0142)
# knee_MTR.setup("0")

# test = RMD_X8(0x0141)
# test.setup("0")

def run():
    while 1:
        # data2 = shoulder_MTR.on_n_set_id(1,1)
        # data2 = shoulder_MTR.write_motor_zero_rom()
        # print("{:.2f}\t {:.2f}".format(data1, data2))
        # print("{:.2f}".format(data1))
        # print(data2)

        data1 = shoulder_MTR.torque_closed_loop(0)["POS"] * 360 / 65535
        print(data1)

def worker(num):
    test = RMD_X8(0x0140+num)
    test.setup("0")
    time.sleep(0.01)
    test.running_command()
    time.sleep(0.01)
    test.torque_closed_loop(0)["POS"] * 360 / 65535
    time.sleep(0.01)
 
def super_scan():
    threads = []
    for i in range(1, 32):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        time.sleep(1)
    
    for t in threads:
        t.join()

def super_duper_scan():
    START_ID = 0
    END_ID = 32

    # Create a CAN bus instance
    # bus = can.interface.Bus(bustype='socketcan', channel='can8')

    # Scan for active nodes on the bus
    active_nodes = []
    for i in range(START_ID, END_ID + 1):
        start = time.time()
        try:
            test = RMD_X8(0x0140+i)
            test.setup("0")
            response = test.running_command()
            test.torque_closed_loop(0)["POS"] * 360 / 65535
            print(response)

            if response is not None:
                active_nodes.append(i)
                print(f"Node {i} is active")    
            
        except:
            pass


    print("Bus scanning complete")
    print(f"Active nodes: {active_nodes}")
    
if __name__ == "__main__":
    print(shoulder_MTR.write_encoder_zero_rom())
    # print(shoulder_MTR.write_motor_zero_rom())
    
    # run()
    