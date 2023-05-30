# RMD-X8 Python Library
# Copyright 2022 Sanjay Sunil

import can
import os
import time
import numpy as np
import ctypes

class RMD_X8:
    """
    A class to read and write on the RMD-X8 motor.

    ...

    Attributes
    ----------
    bus : type
        the can bus channel used to communicate with the motor
    identifier : type
        the motor's identifier on the can bus

    Methods
    -------
    setup():
        Setup the can bus connection.
    send_cmd(data, delay):
        Send a frame data to the motor.
    read_pid():
        Read the motor's current PID parameters.
    write_pid_ram(data):
        Write PID parameters to the RAM.
    write_pid_rom(data):
        Write PID parameters to the ROM.
    read_acceleration():
        Read the motor's acceleration data.
    write_acceleration_ram(data):
        Write the acceleration to the RAM of the motor.
    read_encoder():
        Read the current position of the encoder.
    write_encoder_offset(data):
        Set the motor's encoder offset.
    write_motor_zero_rom():
        Write the current position of the motor to the ROM 
        as the motor zero position.
    read_multi_turns_angle():
        Read the multi-turn angle of the motor.
    read_single_turn_angle():
        Read the single circle angle of the motor.
    motor_off():
        Turn off the motor, while clearing the motor operating 
        status and previously received control commands.
    motor_stop():
        Stop the motor, but do not clear the operating state and 
        previously received control commands.
    motor_running():
        Resume motor operation from the motor stop command.
    read_motor_status_1():
        Reads the motor's error status, voltage, temperature and 
        other information. 
    read_motor_status_2():
        Reads the motor temperature, voltage, speed and encoder 
        position.
    read_motor_status_3():
        Reads the phase current status data of the motor.
    clear_motor_error_flag():
        Clears the error status of the currrent motor.
    torque_closed_loop(data):
        Control torque current output of the motor.
    speed_closed_loop(data):
        Control the speed of the motor.
    position_closed_loop_1(data):
        Control the position of the motor (multi-turn angle).
    position_closed_loop_2(data):
        Control the position of the motor (multi-turn angle).
    position_closed_loop_3(data):
        Control the position of the motor (single-turn angle).
    position_closed_loop_4(data):
        Control the position of the motor (single-turn angle).
    """

    def __init__(self, identifier):
        """
        Constructs all the necessary attributes for the RMDX8 object.
        """
        self.bus = None
        self.identifier = identifier
        self.delay = 0.0001

    def setup(self,channel):
        """
        Setup the can bus connection.

        Returns
        -------
        self.bus : type
            The bus used to communicate with the motor.
        """
        try:
            os.system("sudo /sbin/ip link set can{} up type can bitrate 1000000".format(channel))
            time.sleep(0.001)
        except Exception as e:
            print(e)

        try:
            bus = can.interface.Bus(bustype='socketcan', channel='can{}'.format(channel) ,bitrate = 1000000, tx_ack=True)

        except OSError:
            print('err: PiCAN board was not found.')
            exit()
        except Exception as e:
            print(e)

        self.bus = bus
        return self.bus

    def send_cmd(self, data, delay):
        """
        Send frame data to the motor.

        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.
        delay : int/float
            The time to wait after sending data to the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = can.Message(arbitration_id=self.identifier, data=data, is_extended_id=False)
        self.bus.send(message)
        time.sleep(self.delay)

        filter = [{"can_id": self.identifier, "can_mask": 0xFFF}]
        self.bus.set_filters(filter)

        received_message = self.bus.recv(timeout=0.1)
        return received_message

    def read_pid(self):
        """
        Read the motor's current PID parameters.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        msg = self.send_cmd(message, self.delay)

        data = [x for x in msg.data]
        data = [hex(msg.data[i]) for i in range(len(msg.data))]

        for i in range(len(data)):
            if len(data[i]) == 3:
                data[i] = "0" + data[i][2:]
            else:
                data[i] = data[i][2:]
            
        output = [data[2], data[3], data[4], data[5], data[6], data[7]]
        
        int16_value = [int(output[i],16) for i in range(len(output))]
        
        result = {"POS_Kp":int16_value[0],
                  "POS_Ki":int16_value[1],
                  "SPD_Kp":int16_value[2],
                  "SPD_Ki":int16_value[3],
                  "TRQ_Kp":int16_value[4],
                  "TRQ_Ki":int16_value[5]}
        return result   

    def write_pid_ram(self, data):
        """
        Write PID parameters to the RAM.

        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x31, 0x00, data[0], data[1],
                   data[2], data[3], data[4], data[5]]
        return self.send_cmd(message, self.delay)

    def write_pid_rom(self, data):
        """
        Write PID parameters to the ROM.

        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x32, 0x00, data[0], data[1], data[2], data[3], data[4], data[5]]
        return self.send_cmd(message, self.delay)

    def read_acceleration(self):
        """
        Read the motor's acceleration data.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def write_acceleration_ram(self, data):
        """
        Write the acceleration to the RAM of the motor.

        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x34, 0x00, 0x00, 0x00,
                   data[0], data[1], data[2], data[3]]
        return self.send_cmd(message, self.delay)

    def read_encoder(self):
        """
        Read the current position of the encoder.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        msg = self.send_cmd(message, self.delay)

        data = [x for x in msg.data]
        data = [hex(msg.data[i]) for i in range(2,len(msg.data))]
        output = ["0","0"]

        output[0] = data[1][2:] + data[0][2:]
        output[1] = data[3][2:] + data[2][2:]
        output = [int(str(output[i]),16) for i in range(len(output))]
        
        result = {"pos":output[0],
                "OGpos":output[1]}
        return result

    def write_encoder_offset(self, data):
        """
        Set the motor's encoder offset.

        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        data = self.DECIMAL_2_16HEX(data,2)
        print(data)
        message = [0x91, 0x00, 0x00, 0x00,
                   0x00, 0x00, data[0], data[1]]
        return self.send_cmd(message, self.delay)

    def write_motor_zero_rom(self):
        """
        Write the current position of the motor to the ROM as the motor zero position.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x19, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def write_encoder_zero_rom(self):
        """
        Write encoder values to ROM as motor zero command.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x91, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)
    
    def read_multi_turns_angle(self):
        """
        Read the multi-turn angle of the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x92, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        msg = self.send_cmd(message, self.delay)

        data = [x for x in msg.data]
        data = [hex(msg.data[i]) for i in range(len(msg.data))]

        for i in range(len(data)):
            if len(data[i]) == 3:
                data[i] = "0" + data[i][2:]
            else:
                data[i] = data[i][2:]
            
        byte = data[7] + data[6] + data[5] + data[4] + data[3] + data[2] + data[1]
        
        byte = int(byte, 16)
        if byte >= 0x80000000000000:
            byte = -((~byte) & (72057594037927935))
        
        int64 = ctypes.c_int64(byte).value

        result = {"MulPOS":int64}
        return result

    def read_single_turn_angle(self):
        """
        Read the single circle angle of the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x94, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        msg = self.send_cmd(message, self.delay)
        data = self.read_msg(msg)
        output = {"Angle": data["Pos"]}
        return output

    def motor_off(self):
        """
        Turn off the motor, while clearing the motor operating status and previously received control commands.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x80, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def motor_stop(self):
        """
        Stop the motor, but do not clear the operating state and previously received control commands.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x81, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def motor_run(self):
        """
        Resume motor operation from the motor stop command.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x88, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def read_motor_status_1(self):
        """
        Reads the motor's error status, voltage, temperature and other information.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x9A, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def read_motor_status_2(self):
        """
        Reads the motor temperature, voltage, speed and encoder position.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x9C, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        msg = self.send_cmd(message, self.delay)

        data = [x for x in msg.data]
        data = [hex(msg.data[i]) for i in range(len(msg.data))]

        for i in range(len(data)):
            if len(data[i]) == 3:
                data[i] = "0" + data[i][2:]
            else:
                data[i] = data[i][2:]
            
        output = [data[3]+data[2], data[5]+data[4], data[7]+data[6]]

        int16_value = [int(output[i],16) for i in range(len(output))]
        uint16_value = [np.uint16(int16_value[i]) for i in range(len(int16_value))]
        
        torque_value = self.hex_to_int16(output[0])
        speed_value = self.hex_to_int16(output[1])

        result = {"TRQ":torque_value,
                  "SPD":speed_value,
                  "POS":uint16_value[2]}
        return result     

    def read_motor_status_3(self):
        """
        Reads the phase current status data of the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x9D, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def clear_motor_error_flag(self):
        """
        Clears the error status of the currrent motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0x9B, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def torque_closed_loop(self, data):
        """
        Control torque current output of the motor.        
        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        hexData = self.DECIMAL_2_16HEX(data,2)
        message = [0xA1, 0x00, 0x00, 0x00, hexData[0], hexData[1], 0x00, 0x00];
        msg = self.send_cmd(message, self.delay)

        data = [x for x in msg.data]
        data = [hex(msg.data[i]) for i in range(len(msg.data))]

        for i in range(len(data)):
            if len(data[i]) == 3:
                data[i] = "0" + data[i][2:]
            else:
                data[i] = data[i][2:]
            
        output = [data[3]+data[2], data[5]+data[4], data[7]+data[6]]

        temp_value   = ctypes.c_int8(int(data[1], 16)).value
        torque_value = self.hex_to_int16(output[0])
        speed_value  = self.hex_to_int16(output[1])
        pos_value    = ctypes.c_uint16(self.hex_to_int16(output[2])).value


        result = {"TMP":temp_value,
                  "TRQ":torque_value,
                  "SPD":speed_value,
                  "POS":pos_value}
        return result        

    def speed_closed_loop(self, data):
        """
        Control the speed of the motor.

        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.
            speed control low byte
            speed control
            speed control
            speed control high byte

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        hexData = self.DECIMAL_2_16HEX(data, 4)
        message = [0xA2, 0x00, 0x00, 0x00,
                   hexData[0], hexData[1], hexData[2], hexData[3]]
        msg = self.send_cmd(message, self.delay)
        
        return self.read_msg(msg)

    def position_closed_loop_1(self, data):
        """
        Control the position of the motor (multi-turn angle) 360+.
        [posL,pos,pos,posH]

        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        hexData = self.DECIMAL_2_16HEX(data * 900,4)
        message = [0xA3, 0x00, 0x00, 0x00, hexData[0], hexData[1], hexData[2], hexData[3]]
        msg = self.send_cmd(message, self.delay)
        
        return self.read_msg(msg)

    def position_closed_loop_2(self, spd, pos):
        """
        Control the position of the motor (multi-turn angle) 360+.
        [speedL, speedH, posL, pos, pos, posH]
        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        Hexspd = self.DECIMAL_2_16HEX(spd,2)
        Hexpos = self.DECIMAL_2_16HEX(pos * 900, 4)

        message = [0xA4, 0x00, Hexspd[0], Hexspd[1], Hexpos[0], Hexpos[1], Hexpos[2], Hexpos[3]]
        msg = self.send_cmd(message, self.delay)
        if msg == 0:
            return

        data = [x for x in msg.data]
        data = [hex(msg.data[i]) for i in range(len(msg.data))]

        for i in range(len(data)):
            if len(data[i]) == 3:
                data[i] = "0" + data[i][2:]
            else:
                data[i] = data[i][2:]
            
        output = [data[3]+data[2], data[5]+data[4], data[7]+data[6]]
        
        torque_value = self.hex_to_int16(output[0])
        speed_value = self.hex_to_int16(output[1])
        pos_value    = ctypes.c_uint16(self.hex_to_int16(output[2])).value

        result = {"TRQ":torque_value,
                  "SPD":speed_value,
                  "POS":pos_value}
        return result   

    def position_closed_loop_3(self, data):
        """
        Control the position of the motor (single-turn angle) 360 degree.
        [direction, XX, XX, pos, pos, XX, XX]

        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        message = [0xA5, data[0], 0x00, 0x00, data[1], data[2], 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def position_closed_loop_4(self, spn, spd, pos):
        """
        Control the position of the motor (single-turn angle) 360 degree.
        [direction, speedL, speedH, posL, posH, XX, XX]

        Parameters
        ----------
        data : list
            The frame data to be sent to the motor.
            DIR = 0x00~0x01
            SPD = UINT16
            POS = UINT16 0~35999

        Returns
        -------
        received_message : list
            Frame data received from the motor after receiving the command.
        """
        Hexspn = self.DECIMAL_2_16HEX(spn,1)
        Hexspd = self.DECIMAL_2_16HEX(spd,2)
        Hexpos = self.DECIMAL_2_16HEX(pos,2)
        #print(hex(Hexpos[1])[2:]+hex(Hexpos[0])[2:])

        message = [0xA6, Hexspn[0], Hexspd[0], Hexspd[1], Hexpos[0], Hexpos[1], 0x00, 0x00]
        msg = self.send_cmd(message, self.delay)

        data = [x for x in msg.data]
        data = [hex(msg.data[i]) for i in range(len(msg.data))]

        for i in range(len(data)):
            if len(data[i]) == 3:
                data[i] = "0" + data[i][2:]
            else:
                data[i] = data[i][2:]
            
        output = [data[3]+data[2], data[5]+data[4], data[7]+data[6]]

        int16_value = [int(output[i],16) for i in range(len(output))]
        output = [np.uint16(int16_value[i]) for i in range(len(int16_value))]
        
        result = {"TRQ":output[0],
                  "SPD":output[1],
                  "POS":output[2]}
        return result

    def DECIMAL_2_16HEX(self,data,byte):
        HexData = hex((data + (1 << 32)) % (1 << 32))
        output = ["0","0","0","0","0","0"]
        result = ["0","0","0","0","0","0"]

        if len(HexData)%2 == 0:
            output = [HexData[-i:-i+2] for i in range(-len(HexData)+2, 0, 2)]
        else:
            output = [HexData[-i:-i+2] for i in range(-len(HexData)+2, 0, 2)]
            output[-1] = output[-1].replace("x", "0")

        output = [int(str(output[i]),16) for i in range(len(output))]
        for i in range(byte):
            try:
                result[i] = output[i] 
            except:
                result[i] = 0
        return(result)

    def read_msg(self, msg):

        """
        temp     : int8   — [     0 : 255]
        torque   : int16  — [-32768 : 32767]
        speed    : int16  — [-32768 : 32767]
        position : uint16 — [     0 : 65535]
        """
        data = [x for x in msg.data]
        data = [hex(msg.data[i]) for i in range(len(msg.data))]
        byte = ["0","0","0","0"]

        for i in range(len(data)):
            if len(data[i]) == 3:
                data[i] = "0" + data[i][2:]
            else:
                data[i] = data[i][2:]

        byte[0] = data[1]
        byte[1] = data[3] + data[2]
        byte[2] = data[5] + data[4]
        byte[3] = data[7] + data[6]

        output = [int(byte[i],16) for i in range(len(byte))]

        result = {  "Temp":output[0],
                    "Torque":output[1],
                    "Speed":output[2],
                    "Pos":output[3]}
        return result

        import struct

    def hex_to_int16(self, hex_str):
        int16 = int(hex_str, 16)
        if int16 & 0x8000:
            int16 = -((int16 ^ 0xFFFF) + 1)
        return int16

    def on_n_set_id(self, ReadWriteFlag, ID):
        """
        Setting motors CAN ID. 1~32
        ReadWriteFlag: 0 -- Write, 1 -- Read
        
        -------
        Send    : [0x79   0x00    ReadWriteFlag   0x00    canID_1     canID_2     canID_3     canID_4]
        Receive : [0x79   0x00    ReadWriteFlag   0x00    canID_1     canID_2     canID_3     canID_4]
        """
        canID = self.DECIMAL_2_16HEX(ID, 4)
        message = [0x79, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

    def running_command(self):
        message = [0x88, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        return self.send_cmd(message, self.delay)

