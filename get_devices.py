import socket

def listen_for_devices(host, port):
    devices = set()  # Use a set to store unique device addresses

    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()

    print(f"Listening for devices on {host}:{port}...")

    try:
        server_socket.settimeout(10)  # Set a timeout of 10 seconds
        while True:
            try:
                # Accept incoming connection
                client_socket, client_address = server_socket.accept()
                print(f"Device connected: {client_address}")
                devices.add(client_address[0])  # Add device IP to the set of devices
                client_socket.close()
            except socket.timeout:
                print("No more devices connecting. Stopping server.")
                break
    except KeyboardInterrupt:
        print("Server stopped.")

    return devices

if __name__ == "__main__":
    host = '192.168.1.100'  # Your specific IP address
    port = 10000  # Your specific port
    devices = listen_for_devices(host, port)
    print("Connected devices:", devices)
