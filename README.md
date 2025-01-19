# SeismoBug P Application (SBP)

The SeismoBug P Application (SBP) is a Python-based tool designed to interface with SeismoBug P devices, facilitating data acquisition, processing, and visualization for seismic analysis.

## Features

- **Device Communication**: Establishes and manages connections with SeismoBug P devices for real-time data streaming.
- **Data Processing**: Includes scripts for processing and analyzing seismic data, such as `postprocess_T7.py` and `oma.py`.
- **Visualization**: Provides tools for plotting and visualizing seismic data, including Jupyter notebooks like `Plot_single.ipynb`.
- **User Interface**: Offers a graphical user interface for ease of use, implemented in `mainUI.py`.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/iammix/sbp.git
# Navigate to the Project Directory:

```bash
cd sbp
```

## Install Dependencies:

Ensure you have Python installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```
Usage

Running the Main Application:

Execute the main script to start the application:

```bash
python main.py
```

## Project Structure
```bash
SBP/
├── Cutter/                     # Contains tools for data segmentation
├── Data/                       # Directory for storing raw and processed data
├── Records/                    # Saved seismic records
├── sbpproject/                 # Core application modules
├── ui/                         # User interface components
├── main.py                     # Entry point for the application
├── mainUI.py                   # Script for launching the GUI
├── postprocess_T7.py           # Script for processing T7 data
├── oma.py                      # Operational Modal Analysis script
├── Plot_single.ipynb           # Notebook for single data plotting
└── requirements.txt            # List of required Python packages
```
# Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgements
Special thanks to the contributors and the open-source community for their invaluable support.