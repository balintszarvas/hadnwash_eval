import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_against_time(path):
    """
    Read a CSV file from the given path and plot each column (except the first) against the first column (time).
    """
    df = pd.read_csv(path)
    time = df.iloc[:, 0]
    for col in df.columns[1:]:
        plt.plot(time, df[col], label=col)
    plt.xlabel(df.columns[0])
    plt.ylabel('Value')
    plt.title(f'CSV Data from {path}')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_csv_against_time("/Users/balints/Documents/CLS/MLQS/hadnwash_eval/raw_data/two/Linear Accelerometer.csv")