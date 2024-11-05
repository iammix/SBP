import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QComboBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.fft import fft
from scipy.signal import butter, filtfilt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Acceleration Data Analyzer")
        self.setGeometry(100, 100, 800, 600)

        # Initialize widgets
        self.button_load = QPushButton('Load CSV', self)
        self.button_load.clicked.connect(self.load_csv)

        self.device_selector = QComboBox(self)  # Dropdown menu for devices
        self.device_selector.currentIndexChanged.connect(self.device_changed)

        self.button_plot = QPushButton('Plot Acceleration', self)
        self.button_plot.clicked.connect(self.plot_data)

        self.button_fft = QPushButton('FFT Acceleration', self)
        self.button_fft.clicked.connect(self.apply_fft)

        self.button_low_pass = QPushButton('Low Pass Filter', self)
        self.button_low_pass.clicked.connect(lambda: self.apply_filter('low'))

        # Setup layout
        layout = QVBoxLayout()
        layout.addWidget(self.button_load)
        layout.addWidget(self.device_selector)
        layout.addWidget(self.button_plot)
        layout.addWidget(self.button_fft)
        layout.addWidget(self.button_low_pass)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Data storage
        self.data = None
        self.selected_device = None

    def load_csv(self):
        # Open a file dialog to load the CSV file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)", options=options)
        if file_name:
            # Read the CSV starting from the correct data line
            self.data = pd.read_csv(file_name, skiprows=36)
            print("Data Loaded")
            self.populate_device_selector()

    def populate_device_selector(self):
        """Populate the dropdown menu with device channels (e.g., '57028:ch1', '57028:ch2')."""
        if self.data is not None:
            device_columns = [col for col in self.data.columns if ':' in col]  # Extract columns with devices
            self.device_selector.addItems(device_columns)
            self.selected_device = device_columns[0]  # Default to the first device

    def device_changed(self, index):
        """Handle the device selection change."""
        self.selected_device = self.device_selector.currentText()

    def plot_data(self):
        if self.data is not None and self.selected_device:
            plt.figure()
            plt.plot(self.data['Time'], self.data[self.selected_device])
            plt.title(f'Acceleration for {self.selected_device}')
            plt.xlabel('Time')
            plt.ylabel('Acceleration')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def apply_fft(self):
        if self.data is not None and self.selected_device:
            acc_data = self.data[self.selected_device].values
            fft_data = fft(acc_data)
            plt.figure()
            plt.plot(np.abs(fft_data))
            plt.title(f'FFT of {self.selected_device} Acceleration')
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude')
            plt.show()

    def apply_filter(self, filter_type):
        if self.data is not None and self.selected_device:
            acc_data = self.data[self.selected_device].values
            b, a = butter(4, 0.1, btype=filter_type)
            filtered_data = filtfilt(b, a, acc_data)
            plt.figure()
            plt.plot(self.data['Time'], filtered_data)
            plt.title(f'{filter_type.capitalize()} Filter on {self.selected_device}')
            plt.xlabel('Time')
            plt.ylabel('Filtered Acceleration')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Run the app
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
