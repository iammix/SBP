import time
import pywifi
from pywifi import const
import socket
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import keyboard
from SBP_device_handler import SeismoBugP

def main():
    node = SeismoBugP()
    node.battery()

if __name__ == '__main__':
    main()