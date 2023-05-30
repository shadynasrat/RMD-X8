from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import threading

class OpenGLThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def __init__(self):
        self.L1 = 22
        self.L2 = 24
        self.H = 50
        self.x = -0
        self.y = -20
        self.theta1 = -11
        self.theta2 = 10
        self.key = b's'
    
    def run(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 800)
        glutCreateWindow(b"Robotic Leg")

        # Set up view
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-40, 40, -40, 40, -40, 40)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        glutDisplayFunc(self.display)
        glutKeyboardFunc(self.keyboard)
        glutMainLoop()

    def display(self):
        print("display")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Draw fixed slider
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, -self.H, 0)
        glEnd()

        #self.update_position()
        #self.inverse_kinematics(self.x, self.y)
        
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
        
        glutSwapBuffers()

# Create and start the OpenGL thread
opengl_thread = OpenGLThread()
opengl_thread.start()

# Wait for the OpenGL thread to finish
opengl_thread.join()
