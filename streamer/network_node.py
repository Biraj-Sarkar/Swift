import socket
import json
import pickle
import time
import torch
import os
import subprocess

try:
    import psutil
except ImportError:
    psutil = None

class SwiftNetworkNode:
    """
    Base class for Swift networking.
    Handles the transmission of Bitstreams and Telemetry (Buffer, GPU, Battery).
    """
    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port

    def send_data(self, sock, data):
        serialized_data = pickle.dumps(data)
        # Send length first, then data
        sock.sendall(len(serialized_data).to_bytes(4, byteorder='big'))
        sock.sendall(serialized_data)

    def receive_data(self, sock):
        raw_msglen = self._recvall(sock, 4)
        if not raw_msglen: return None
        msglen = int.from_bytes(raw_msglen, byteorder='big')
        return pickle.loads(self._recvall(sock, msglen))

    def _recvall(self, sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet: return None
            data.extend(packet)
        return data

class SwiftServer(SwiftNetworkNode):
    """The 'Server' running on a powerful machine (Encoder)."""
    def start(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        print(f"Server listening on {self.host}:{self.port}...")

        conn, addr = self.server_sock.accept()
        print(f"Connected by {addr}")
        return conn

class SwiftClient(SwiftNetworkNode):
    """The 'Client' running on the device (Decoder)."""
    def __init__(self, host='127.0.0.1', port=5000):
        super().__init__(host, port)
        self.last_measured_latency = 20.0 # ms
        self.buffer_seconds = 0.0

    def connect(self):
        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sock.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")
        return self.client_sock

    def update_local_metrics(self, gpu_latency, buffer):
        """Update metrics that are measured internally by the decoder loop."""
        self.last_measured_latency = gpu_latency
        self.buffer_seconds = buffer

    def _get_battery_level(self):
        """Probes the actual system battery level."""
        if psutil:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent

        # Fallback for Windows/Linux if psutil fails or isn't installed
        try:
            if os.name == 'nt': # Windows
                output = subprocess.check_output("WMIC Path Win32_Battery Get EstimatedChargeRemaining", shell=True)
                return int(output.decode().split()[1])
            else: # Linux
                output = subprocess.check_output("cat /sys/class/power_supply/BAT0/capacity", shell=True)
                return int(output.decode().strip())
        except:
            return 100 # Default if probing fails

    def _get_gpu_load(self):
        """Probes actual GPU utilization if available (NVIDIA)."""
        try:
            output = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True)
            return int(output.decode().strip())
        except:
            return 0

    def get_hardware_telemetry(self):
        """
        Returns real-time hardware telemetry by probing the device.
        """
        return {
            'gpu_latency': self.last_measured_latency,
            'gpu_load': self._get_gpu_load(),
            'battery_level': self._get_battery_level(),
            'buffer_seconds': self.buffer_seconds
        }
