import time
import pywifi
from pywifi import const
import socket
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import keyboard
from datetime import datetime, timedelta
import zlib
import pandas as pd

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
    def __init__(self, sbp_name=None, ip='192.168.1.100', port=10000):
        self.sbp_name = sbp_name
        self.ip = ip
        self.port = port
        self._device_name = None
        self._firmware = None
        self._color = None
        self._alias = None
        self._packettype = None
        self._packetsize = None
        self._sps = None
        self._packetid = None
        self._cmd = None
        self._ticks_1000MS = 60000
        if sbp_name is None:
            self.info()

    def __sbp_network(self):
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]
        iface.scan()
        time.sleep(5)
        network_list = iface.scan_results()
        self.net_list = []
        for network in network_list:
            if "SBP_" in network.ssid:
                self.net_list.append(network.ssid)
        target_network = self.sbp_name
        if target_network == None:
            raise ValueError("sbp_name is not defined in the class constructor")

        profile = pywifi.Profile()
        profile.ssid = target_network
        profile.auth = const.AUTH_ALG_OPEN
        tmp_profile = iface.add_network_profile(profile)
        iface.connect(tmp_profile)
        time.sleep(5)
        if iface.status() == const.IFACE_CONNECTED:
            print(f"Connected to {target_network}")
        else:
            print("Connection Failed")


    def configure(self, ssid=None, password=None, server=None, port=None, alias=None, color=None):
        server_ip = '192.168.1.1'
        server_port = 6666
        self.__sbp_network()
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((server_ip, server_port))
            if ssid is not None:
                command = f"ssid {ssid}"
                client_socket.sendall(command.encode())
                time.sleep(3)
            if password is not None:
                command = f"pass {password}"
                client_socket.sendall(command.encode())
                time.sleep(3)
            if server is not None:
                command = f"server {server}"
                client_socket.sendall(command.encode())
                time.sleep(3)
            if port is not None:
                command = f"port {port}"
                client_socket.sendall(command.encode())
                time.sleep(3)
            if alias is not None:
                command = f"alias {alias}"
                client_socket.sendall(command.encode())
                time.sleep(3)
            if color is not None:
                command = f"color {color}"
                client_socket.sendall(command.encode())
                time.sleep(3)
            time.sleep(1)
        finally:
            command = 'restart'
            client_socket.sendall(command.encode())
            time.sleep(1)
            client_socket.close()

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
        self._firmware = response.split('\n')[3].split(' ')[1]
        self._color = response.split('\n')[4].split(' ')[1]
        self._alias = response.split('\n')[5].split(' ')[1]
        self._packettype = response.split('\n')[6].split(' ')[1]
        self._packetsize = response.split('\n')[7].split(' ')[1]
        self._sps = response.split('\n')[8].split(' ')[1]
        self._packetid = response.split('\n')[9].split(' ')[1]



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
        return self._battery.split('%')[0]

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

    def _update_plot(self, frame):
        self.sline.set_data(self.t, self.a*1000)
        return self.sline,

    def _parse_packet(self, packet):
        packetsize = (len(packet)-10-12-4-8)//9
        lat = int.from_bytes(packet[-24:-20], 'big')
        lng = int.from_bytes(packet[-20:-16], 'big')
        alt = int.from_bytes(packet[-16:-14], 'big')
        vel = int(packet[-14])
        rssi = int(packet[-13])
        crc32 = int.from_bytes(packet[-12:-8], 'big')
        crc32_calc = zlib.crc32(packet[:-12]) & 0xffffffff
        ID = packet[-8:-2].decode('utf-8')
        num = int.from_bytes(packet[-2:], 'big')
        start = datetime.fromtimestamp(int.from_bytes(packet[0:4], 'big')) + timedelta(seconds=int.from_bytes(packet[4:6], 'big')/self._ticks_1000MS)
        dur = timedelta(seconds=int.from_bytes(packet[6:10], 'big')/self._ticks_1000MS)
        step = dur / packetsize
        t = np.array([start + i * step for i in range(1, packetsize+1)])
        a = np.zeros((packetsize, 3))
        for col in range(3):
            line = 0
            for j in range(10+3*col, len(packet)-4, 9):
                val = (packet[j]<<12) | (packet[j+1]<<4) |(packet[j+2]>>4)
                if val >= 524288:
                    val -= 1048576
                try:
                    a[line][col] = val/256000
                    line +=1
                except:
                    pass
        return lat, lng, alt, vel, rssi, crc32, crc32_calc, ID, num, t, a, start, packetsize, dur


    def start(self, savedata=False, terminate_value=10):
        command = 'start'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        client_socket.send(command.encode())
        time.sleep(4)
        running = True
        count = 0

        
        all_t = np.empty(0)
        all_a = np.empty((0,3))
        stats_packetbytes = np.empty(0).astype(int)
        stats_packetsize = np.empty(0).astype(int)
        stats_start = np.empty(0)
        stats_dur = np.empty(0)
        stats_gap = np.append(np.empty(0), 0)
        stats_sysdatetime = np.empty(0)
        stats_crc32 = np.empty(0).astype(int)
        stats_crc32_calc = np.empty(0).astype(int)
        stats_lat = np.empty(0).astype(float)
        stats_lng = np.empty(0).astype(float)
        stats_alt = np.empty(0).astype(int)
        stats_vel = np.empty(0).astype(int)
        stats_rssi = np.empty(0).astype(int)
        stats_ID = np.empty(0).astype(str)
        stats_num = np.empty(0).astype(int)
        print('Start streaming . . .')
        try:
            while running:
                try:
                    packet = client_socket.recv((10 + 9 * int(self._packetsize) + 12 + 4 + 8))
                    lat, lng, alt, vel, rssi, crc32, crc32_calc, ID, num, t, a, start, packetsize, dur = self._parse_packet(packet)
                    packet_info = f"Packet : {ID} {num:03d}, size : {len(packet)}, time : {start.strftime('%d/%m/%Y %H:%M:%S.%f')}, CRC32 read : {crc32:08X}, CRC32 calc : {crc32_calc:08X}"
                    #print(packet)
                    print(packet_info)
                    # append to total matrices
                    all_t = np.append(all_t, t)
                    all_a = np.vstack((all_a, a))
                    stats_start = np.append(stats_start,start)
                    stats_packetbytes = np.append(stats_packetbytes,(len(packet)))
                    stats_packetsize = np.append(stats_packetsize,packetsize)
                    stats_dur = np.append(stats_dur,dur)
                    gap = (all_t[len(all_t)-packetsize] - all_t[len(all_t)-packetsize-1]).total_seconds()
                    stats_lat = np.append(stats_lat, lat/1e6)
                    stats_lng = np.append(stats_lng, lng/1e6)
                    stats_alt = np.append(stats_alt, alt)
                    stats_vel = np.append(stats_vel, vel)
                    stats_rssi = np.append(stats_rssi, -rssi)
                    stats_crc32 = np.append(stats_crc32, crc32)
                    stats_crc32_calc = np.append(stats_crc32_calc, crc32_calc)
                    stats_ID = np.append(stats_ID, ID)
                    stats_num = np.append(stats_num, num)
                    count += 1
                    time.sleep(1)
                    print(count, terminate_value)
                    if count == terminate_value:
                        running = False
                except:
                    pass
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exiting . . .")

        client_socket.send('stop'.encode())
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,12))
        ax1.plot(all_t, all_a[:, 0])
        ax2.plot(all_t, all_a[:, 1])
        ax3.plot(all_t, all_a[:, 2])
        ax1.set_title('x-axis')
        ax2.set_title('y-axis')
        ax3.set_title('z-axis')
        plt.show()
        client_socket.close()
        
        if savedata:
            all_date = np.array([t.strftime('%d/%m/%Y') for t in all_t])
            all_time = np.array([t.strftime('%H:%M:%S.%f') for t in all_t])
            df = pd.DataFrame(all_date, columns = ['Date'])
            df['Time'] = all_time
            df['Acc x'] = all_a[:,0]
            df['Acc y'] = all_a[:,1]
            df['Acc z'] = all_a[:,2]
            df['Acc x'] = df['Acc x'].apply(lambda x: '{:.6f}'.format(x))
            df['Acc y'] = df['Acc y'].apply(lambda x: '{:.6f}'.format(x))
            df['Acc z'] = df['Acc z'].apply(lambda x: '{:.6f}'.format(x))
            df.to_csv('Record.txt', sep = '\t', index = False)



if __name__ == '__main__':
    node = SeismoBugP()
    node.start(savedata=True, terminate_value=10)