import os
import sys
import matplotlib.pyplot as plt
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtWidgets import QMessageBox, QDialog, QMainWindow
from SBP_device_handler import SeismoBugP
import pywifi
from pywifi import const
import time
import numpy as np
import socket



class ConfigPage(QDialog):
    def __init__(self):
        super(ConfigPage, self).__init__()
        uic.loadUi('./ui/configure_device.ui', self)
        self.search_sbp_btn.clicked.connect(self._get_sbp_from_wifi)
        self.configure_device_btn.clicked.connect(self.__connect_to_wifi)

    def _get_sbp_from_wifi(self):
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]
        iface.scan()
        time.sleep(5)
        network_list = iface.scan_results()
        net_list = []
        for network in network_list:
            if "SBP_" in network.ssid:
                net_list.append(network.ssid)
        self.sbp_list.addItems(np.unique(np.array(net_list)).tolist())

    def __connect_to_wifi(self):
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]
        iface.disconnect()
        time.sleep(1)
        iface.scan()
        # Get the scanned WiFi networks
        networks = iface.scan_results()

        # Find the network with the given SSID
        target_network = None
        for item in self.sbp_list.selectedItems():
            ssid = item.text()
        for network in networks:
            if network.ssid == ssid:
                target_network = network
                break

        # If the target network is found, try to connect
        if target_network is not None:
            # Create a wifi profile
            profile = pywifi.Profile()
            profile.ssid = ssid
            profile.auth = const.AUTH_ALG_OPEN
            # Add the new profile
            tmp_profile = iface.add_network_profile(profile)

            # Connect to the WiFi network
            iface.connect(tmp_profile)

            # Wait for connection to establish
            time.sleep(5)

            # Check if the connection was successful
            if iface.status() == const.IFACE_CONNECTED:
                print("Connected to", ssid)
            else:
                print("Connection failed")
        else:
            print("WiFi network not found")
        self.__configure()

    def __configure(self):
        server_ip = '192.168.1.1'
        port = 6666
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((server_ip, port))

            command = f'ssid {self.ssid_le.text()}'
            client_socket.sendall(command.encode())
            response = client_socket.recv(1024).decode()
            time.sleep(3)

            command = f'pass {self.pass_le.text()}'
            client_socket.sendall(command.encode())
            response = client_socket.recv(1024).decode()
            time.sleep(3)
            
            command = f'server {self.tcp_server_le.text()}'
            client_socket.sendall(command.encode())
            response = client_socket.recv(1024).decode()
            time.sleep(3)
            
            command = 'restart'
            client_socket.sendall(command.encode())

        finally:
            client_socket.close()


class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        uic.loadUi('./ui/main.ui', self)
        self.get_active_devices_btn.clicked.connect(self._get_active_devices)

        self.get_device_info_btn.clicked.connect(self._get_device_info)
        self.actionRegister_Devices.triggered.connect(self.go_to_device_config)
    
    def go_to_device_config(self):
        self.gotoconfig = ConfigPage()
        self.gotoconfig.show()

    def _get_active_devices(self):
        command = 'list'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        client_socket.send(command.encode())
        client_socket.send(command.encode())
        time.sleep(10)
        response = client_socket.recv(1024).decode()
        print(response)
        
    def _get_device_info(self):
        self.device = SeismoBugP()
        self.device.info()
        self.device_name_lbl.setText(self.device._device_name)
        self.device_firmware_lbl.setText(self.device._firmware)
        self.device_color_lbl.setText(self.device._color)
        self.device_alias_lbl.setText(self.device._alias)
        self.device_packet_type_lbl.setText(self.device._packettype)
        self.device_packet_size_lbl.setText(self.device._packetsize)
        self.device_sps_lbl.setText(self.device._sps)
        self.battery_bar.setValue(int(self.device.battery()))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainUI()
    window.show()
    app.exec()
