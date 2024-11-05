import os
import pandas as pd
import typing
import matplotlib.pyplot as plt


DEVICE = {"RISE_1": "57044", "RISE_3": "57030", "RISE_5": "57045", "RISE_7": "57047", "RISE_9": "57046",
    "RISE_11": "57029", "RISE_13": "57028"}

DATA = 'Data\T7'

HEADER = ["Time", "57028:ch1", "57028:ch2", "57028:ch3", "57029:ch1", "57029:ch2", "57030:ch1", "57030:ch2",
          "57044:ch1", "57044:ch2", "57045:ch1", "57045:ch2", "57047:ch2", "57047:ch1", "57046:ch2", "57046:ch1",
          "57045:ch3", "57044:ch3", "57029:ch3"]


def lord_parser(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, skiprows=36, delimiter=';', names=HEADER, header=0)
    # Convert the Time column to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Filter data for the first 10 seconds
    start_time = df['Time'].iloc[0]
    df_10sec = df[df['Time'] <= start_time + pd.Timedelta(seconds=)]

    # Format time to display only HH:MM
    for col in df_10sec.columns[1:]:  # Skip the 'Time' column
        df_10sec[col] = df_10sec[col] - df_10sec[col].mean()
    df_10sec['Time'] = df_10sec['Time'].dt.strftime('%H:%M:%S.%f')

    for col in df_10sec.columns[1:3]:  # Skip the first column (Time)
        plt.figure()
        plt.plot(df_10sec['Time'], df_10sec[col])
        plt.xlabel("Time (HH:MM)")
        plt.ylabel(f"{col} Acceleration")
        plt.title(f"{col} vs Time (First 10 Seconds)")
        plt.xticks(rotation=45)
        plt.show()



def post_process():
    lord_file = os.path.join(DATA, 'lord\\3_t7.csv')
    lord_parser(lord_file)


if __name__ == '__main__':
    post_process()
