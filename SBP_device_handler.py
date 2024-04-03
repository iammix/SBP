import time
import pywifi
from pywifi import const
import socket
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import keyboard


def scan_wifi_networks():
    wifi = pywifi.PyWiFi()
    
    iface = wifi.interfaces()[0]
    iface.scan()
    time.sleep(5)
    network_list = iface.scan_results()
    for network in network_list:
        print(network.ssid)

def handle_client(client_socket):
    try:
        time.sleep(4)
        command = 'battery'
        client_socket.send(command.encode())
        response = client_socket.recv(1024).decode()
        print("Battery Response:", response)
    finally:
        client_socket.close()



def battery_command():
    command = 'battery'
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('192.168.1.100', 10000))
    server_socket.listen()
    device_no = 2
    count = 0
    while device_no > count:
        client_socket, client_address = server_socket.accept()
        print("connection from:", client_address)
        client_thread = threading.Thread(target=handle_client, args=[client_socket])
        client_thread.start()
        count += 1


class SeismoBugP:
    def __init__(self):
        self._device_name = None
        self._firmware = None
        self._color = None
        self._alias = None
        self._packettype = None
        self._packetsize = None
        self._sps = None
        self._packetid = None
        self._cmd = None
        self.info()

    def configure(self):
        pass

    def __exec_cmd(self, command):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        client_socket.send(command.encode())
        time.sleep(4)
        response = client_socket.recv(1024).decode()
        return response


    def info(self):
        command = 'info'
        response = self.__exec_cmd(command)
        self._device_name = response.split('\n')[0]
        self._firmware = response.split('\n')[4].split(' ')[1]
        self._color = response.split('\n')[5].split(' ')[1]
        self._alias = response.split('\n')[6].split(' ')[1]
        self._packettype = response.split('\n')[7].split(' ')[1]
        self._packetsize = response.split('\n')[8].split(' ')[1]
        self._sps = response.split('\n')[9].split(' ')[1]
        self._packetid = response.split('\n')[10].split(' ')[1]



    @property
    def packettype(self):
        command = 'packettype'
        response = self.__exec_cmd(command)
        self._packettype = response.split('\n')[4]
        return self._packettype

    @packettype.setter
    def packettype(self, command):
        self._cmd = command
        command = f'packettype {self._cmd}'
        response = self.__exec_cmd(command)
        self._packettype = response.split('\n')[4]
        
    @property
    def packetsize(self):
        command = 'packetsize'
        response = self.__exec_cmd(command)
        self._packetsize = response.split('\n')[4]
        return self._packetsize
    
    @packetsize.setter
    def packetsize(self, command):
        self._cmd = command
        command = f'packetsize {self._cmd}'
        response = self.__exec_cmd(command)
        self._packtsize = response.split('\n')[4]

    @property
    def sps(self):
        command = 'sps'
        response = self.__exec_cmd(command)
        self._sps = response.split('\n')[4]
        return self._sps

    @sps.setter
    def sps(self, command):
        self._cmd = command
        if self._cmd not in [500, 250, 125, 62.5, 31.25]:
            raise ValueError("SPS must be one of the following values: 500, 250, 125, 62.5, 31.25")
        command = f'sps {self._cmd}'
        response = self.__exec_cmd(command)
        self._sps = response.split('\n')[4]

    @property
    def packetid(self):
        command = 'packetid'
        response = self.__exec_cmd(command)
        self._packetid = response.split('\n')[4]
        return self._packetid

    @packetid.setter
    def packetid(self, command):
        self._cmd = command
        if self._cmd not in ['on', 'off']:
            raise ValueError("Packet ID accepted values are 'on' 'off'")
        command = f'packtid {self._cmd}'
        response = self.__exec_cmd(command)
        self._packtid = response.split('\n')[4]
    
    @property
    def brightness(self):
        command = 'brightness'
        response = self.__exec_cmd(command)
        self._brightness = response.split('\n')[4]
        return self._brightness

    @brightness.setter
    def brightness(self, command):
        self._cmd = command
        if self._cmd not in list(range(1, 101)):
            raise ValueError('Brightness value must be and integer between 1-100')
        command = f'brightness {self._cmd}'
        self.__exec_cmd(command)
        self._brightness = self._cmd

    def invert(self):
        command = 'invert'
        response = self.__exec_cmd(command)
        print('Invert DONE . . .')
    
    def showalias(self):
        command = 'showalias'
        response = self.__exec_cmd(command)
        print(f"Check Node Display: Node Alias {self._alias}")

    def rssi(self):
        command = 'rssi'
        response = self.__exec_cmd(command)
        self._rssi = response.split('\n')[4]
        print(f'Device RSSI is: {self._rssi}dBm')

    def battery(self):
        command = 'battery'
        response = self.__exec_cmd(command)
        self._battery = response.split('\n')[4]
        print(f'Battery Percentage is: {self._battery}')
    
    def sats(self):
        command = 'sats'
        response = self.__exec_cmd(command)
        self._sats = response.split('\n')[4]
        print(f'Number of connected satellites: {self._sats}')
    
    def location(self):
        command = 'location'
        response = self.__exec_cmd(command)
        self._location = response.split('\n')[4]
        print(f'Location: {self._location}')

    def altitude(self):
        command = 'altitude'
        response = self.__exec_cmd(command)
        self._altitude = response.split('\n')[4]
        print(f'Altitude: {self._altitude}')

    def velocity(self):
        command = 'velocity'
        response = self.__exec_cmd(command)
        self._velocity = response.split('\n')[4]
        print(f'Device Velocity: {self._velocity}')

    def hdop(self):
        command = 'hdop'
        response = self.__exec_cmd(command)
        self._hdop = response.split('\n')[4]
        print(f"Device's HDOP: {self._hdop}")
    
    def restart(self):
        command = 'restart'
        self.__exec_cmd(command)
        print('Device RESTARTED . . .')
    
    def coldrestart(self):
        command = 'coldrestart'
        self.__exec_cmd(command)
        print('Device COLDRESTARTED . . .')
    
    def __wait_for_key_press(self):
        print('Press any key to exit . . .')
        keyboard.wait('esc')
    
    def start(self, value=10000):
        command = 'start'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        client_socket.send(command.encode())
        time.sleep(4)
        running = True
        count = 0
        try:
            while True:
                print("Running . . .")
                self.__wait_for_key_press()
                break
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exiting . . .")
        #while running:
        #    response = client_socket.recv(10 + 9 * int(self._packetsize) + 12 + 4 + 8)
        #    lat = int.from_bytes(response[-24:-20], 'big')
        #    lng = int.from_bytes(response[-20:-16], 'big')
        #    alt = int.from_bytes(response[-16:-14], 'big')
        #    vel = int(response[-14])
        #    rssi = int(response[-13])
        #    
        #    for col in range(3):
        #        line = 0
        #        for j in range(10+3*col, len(response)-4, 9):
        #            val = (response[j] << 12) | (response[j+1] << 4) | (response[j+2] >> 4)
        #            if val >= 524288:
        #                val = val - 1048576
        #            try:
        #                a[line][col] = val / 256000
        #                line = line+1 
        #            except:
        #                pass
        #    x=1
        #    
        #    if count == value:
        #        running = False
        #    count += 1
        #    print(lat, lng)
        #    print(rssi)
        #command = 'stop'
        #client_socket.send(command.encode())
        #client_socket.close()