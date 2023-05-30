from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import math
import time
from OpenGL.GLUT import GLUT_KEY_UP, GLUT_KEY_DOWN
import random
from scipy.signal import chirp, spectrogram, gausspulse, sweep_poly
import threading
import matplotlib.pyplot as plt


class SOLO_kinematics():
    def __init__(self, **kwargs):
        self.L1 = 22
        self.L2 = 22
        self.H = 50
        self.x = 0
        self.y = -20
        self.theta1 = 0
        self.theta2 = 0
        self.key = b's'
        self.theta3 = 0
        self.theta4 = 0
        self.theta1_offset = 0
        self.theta2_offset = 0
        self.frequency = 0.1
        self.start = False
        self.exit = False
        self.time = 0
        self.start_time = time.time()
        # self.customs = {'shoulder':{'p_gain':1, 'i_gain':0.1, 'd_gain':0.1}, 'knee':{'p_gain':1, 'i_gain':0.1, 'd_gain':0.1}}
        self.customs = [[0,0,0],[0,0,0]]
        self.index = 0
        self.header = 0
        self.signal= 0
        self.timer = 0
        self.step = 0
        self.make_motion()

    def render(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 800)
        glutCreateWindow(b"Robotic Leg")

        # Set up view
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-40, 40, -40, 40, -40, 40)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity
        glLoadIdentity()
        gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        glutDisplayFunc(self.display)
        glutKeyboardFunc(self.keyboard)
        glutSpecialFunc(self.keyboard)
        glutMainLoop()

    def inverse_kinematics(self):
        d = np.sqrt(self.x**2 + self.y**2)
        phi = np.arctan2(self.y, self.x)
        alpha = np.arccos((self.L1**2 + d**2 - self.L2**2) / (2*self.L1*d))
        beta = np.arccos((self.L2**2 + self.L1**2 - d**2) / (2*self.L1*self.L2))
        self.theta1 = phi + alpha
        self.theta2 = np.pi + beta

    def make_motion(self):
        n_steps = 100000
        sweep_num = 10

        # Define the frequency and amplitude sweeps
        freq_sweep = np.linspace(0.0001, 0.005, int(n_steps/sweep_num))
        amp_sweep = np.linspace(5, 10, int(n_steps/sweep_num))

        # Create the time vector
        t = np.linspace(0, n_steps, n_steps)

        # Initialize the signal
        self.signal = np.zeros_like(t)

        # Create the signal by sweeping through the frequency and amplitude ranges
        for i in range(len(t)):
            freq = freq_sweep[i % len(freq_sweep)]
            amp = amp_sweep[i % len(amp_sweep)]
            self.signal[i] = amp * np.sin(2 * np.pi * freq * t[i])

        # self.signal = self.signal[95000:]
        # Plot the signal
        # plt.plot(t, self.signal)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.title('Legged Robot Sine Wave Sweep')
        # plt.show()

    def update_position_motion(self):
        """
        FOCUS ON THIS FUNCTION
        """
        # t = time.time()
        # f = 0.1 * t
        # self.x  = 20 * math.sin(2 * math.pi * (4 * f - math.cos(t)));
        # self.y  = -20 + 10 * math.sin(2 * math.pi * (4 * f - math.cos(t)));

        self.y = -20 + self.signal[self.step]
        
        glutPostRedisplay()

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Draw fixed slider
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, -self.H, 0)
        glEnd()

        self.update_position_motion()
        self.inverse_kinematics()

        self.theta1_offset = (self.theta1 / math.pi * 180) + 2.6
        self.theta2_offset = (self.theta2 / math.pi * 180) - 150 -60
        x_foot = self.L1*np.cos(self.theta1) + self.L2*np.cos(self.theta1 + self.theta2)
        y_foot = self.L1*np.sin(self.theta1) + self.L2*np.sin(self.theta1 + self.theta2)

        # Draw leg segments
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(4.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(self.L1*np.cos(self.theta1), self.L1*np.sin(self.theta1), 0)
        glEnd()

        # Compute foot position using inverse kinematics
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(4.0)
        glBegin(GL_LINES)
        glVertex3f(self.L1*np.cos(self.theta1), self.L1*np.sin(self.theta1), 0)
        glVertex3f(x_foot, y_foot, 0)
        glEnd()

        # Draw foot
        glColor3f(0.0, 1.0, 0.0)
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glVertex3f(x_foot, y_foot, 0)
        glEnd()
        
        x2_foot = self.L1*np.cos(self.theta3) + self.L2*np.cos(self.theta3 + self.theta4)
        y2_foot = self.L1*np.sin(self.theta3) + self.L2*np.sin(self.theta3 + self.theta4)

        # Draw leg segments
        glColor3f(1, 1, 1)
        glLineWidth(10.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(self.L1*np.cos(self.theta3), self.L1*np.sin(self.theta3), 0)
        glEnd()

        # Compute foot position using inverse kinematics
        glColor3f(1, 1, 1)
        glLineWidth(10.0)
        glBegin(GL_LINES)
        glVertex3f(self.L1*np.cos(self.theta3), self.L1*np.sin(self.theta3), 0)
        glVertex3f(x2_foot, y2_foot, 0)
        glEnd()

        # Draw foot
        glColor3f(1.0, 1.0, 1.0)
        glPointSize(20.0)
        glBegin(GL_POINTS)
        glVertex3f(x2_foot, y2_foot, 0)
        glEnd()
        glutSwapBuffers()
        self.step += 1

    def keyboard(self, key, x, y):
        self.key = key
        if self.key == b'q':
            glutDestroyWindow(b"Robotic Leg")
        elif self.key == b'\x1b':
            self.exit = True
        elif self.key == b'\r':
            self.start = True
        elif self.key == GLUT_KEY_LEFT:
            if self.index > 0:
                self.index -= 1
        elif self.key == GLUT_KEY_RIGHT:
            if self.index < 2:
                self.index += 1
        elif self.key == GLUT_KEY_UP:
            if self.index == 2:
                self.customs[self.header][self.index] += 0.01
            else:
                self.customs[self.header][self.index] += 0.1
        elif self.key == GLUT_KEY_DOWN:
            if self.index == 2:
                self.customs[self.header][self.index] -= 0.01
            else:
                self.customs[self.header][self.index] -= 0.1

        elif self.key == GLUT_KEY_F1:
            self.header = 0
            
        elif self.key == GLUT_KEY_F2:
            self.header = 1


# IK = SOLO_kinematics() 
# thread1 = threading.Thread(target=IK.render)
# thread1.start()