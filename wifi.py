import time
import pywifi
from pywifi import const
import socket
import threading


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
        pass
    
    def configure(self):
        pass


    def packettype(self):
        command = 'packettype'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        print(f"Connected Device: {client_address}")
        client_socket.send(command.encode())
        time.sleep(4)
        response = client_socket.recv(1024).decode()
        self.packettype_value = response.split('\n')[4]
        print(f"Packet Type: {self.packetsize_value}")
    
    def packetsize(self):
        command = 'packetsize'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        print(f"Connected Device: {client_address}")
        client_socket.send(command.encode())
        time.sleep(4)
        response = client_socket.recv(1024).decode()
        self.packetsize_value = response.split('\n')[4]
        print(f"Packet Size: {self.packetsize_value}")

    def battery(self):
        command = 'battery'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        print(f"Connected Device: {client_address}")
        client_socket.send(command.encode())
        time.sleep(4)
        response = client_socket.recv(1024).decode()
        self.battery_value = response.split('\n')[4]
        print(f"Battery: {self.battery_value}")


    def sps(self):
        command = 'sps'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        print(f"Connected Device: {client_address}")
        client_socket.send(command.encode())
        time.sleep(4)
        response = client_socket.recv(1024).decode()
        self.sps_value = response.split('\n')[4]
        print(f"SPS: {self.sps_value}")


    def packetid(self):
        command = 'packetid'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        print(f"Connected Device: {client_address}")
        client_socket.send(command.encode())
        time.sleep(4)
        response = client_socket.recv(1024).decode()
        self.packetid_value = response.split('\n')[4]
        print(f"Packet ID: {self.packetid_value}")
        
    def brightness(self):
        command = 'brightness'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        print(f"Connected Device: {client_address}")
        client_socket.send(command.encode())
        time.sleep(4)
        response = client_socket.recv(1024).decode()
        self.brightness_value = response.split('\n')[4]
        print(f"Brightness: {self.brightness_value}")

    def invert(self):
        command = 'invert'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        print(f"Connected Device: {client_address}")
        client_socket.send(command.encode())
        time.sleep(4)
        print(f"Invert DONE")

    def showalias(self):
        command = 'showalias'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.bind(('192.168.1.100', 10000))
        client_socket.listen()
        client_socket, client_address = client_socket.accept()
        print(f"Connected Device: {client_address}")
        client_socket.send(command.encode())
        print("Check Node")
        time.sleep(4)


        

if __name__ == '__main__':
    node_2 = SeismoBugP()
    node_2.showalias()