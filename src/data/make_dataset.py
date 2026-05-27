# To run .py file in interactive mode
# %%
import os
import pandas as pd
from glob import glob

# ---------------------------------------------------------------
# Set working directory
# ---------------------------------------------------------------

# In interactive mode, the working directory can change each session.
os.chdir(os.path.dirname(__file__))

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyro = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

f = files[0]
data_path = "../../data/raw/MetaMotion\\"

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]  # What the participant is doing (e.g. bench, squat, etc.)
category = f.split("-")[2].rstrip("123")

df = pd.read_csv(f)
df["participant"] = participant
df["label"] = label
df["category"] = category

# --------------------------------------------------------------
# Repeat the process for all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyro_df = pd.DataFrame()

acc_set = 1
gyro_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]  # What the participant is doing (e.g. bench, squat, etc.)
    category = f.split("-")[2].replace("_MetaWear_2019", "").rstrip("123")

    df = pd.read_csv(f)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df], ignore_index=True)

    elif "Gyroscope" in f:
        df["set"] = gyro_set
        gyro_set += 1
        gyro_df = pd.concat([gyro_df, df], ignore_index=True)  # Gyroscope data is sampled at 25Hz, so we will have more rows in this dataset.


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

# Epoch: in Unix time, the number of seconds that have elapsed since 00:00:00 January 1, 1970 (UTC)
acc_df.info()

# The datatype of epoch and time is not in datetime format, so we need to convert it.
# There is one hour difference between the two columns due to summer time.
pd.to_datetime(df["epoch (ms)"], unit="ms")
pd.to_datetime(df["time (01:00)"])

# Set the epoch datetime as the index for time series analysis later on. This will allow us to easily resample the data.
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

# Drop columns
del acc_df["epoch (ms)"], acc_df["time (01:00)"], acc_df["elapsed (s)"]
del gyro_df["epoch (ms)"], gyro_df["time (01:00)"], gyro_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into a function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acc_set = 1
    gyro_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]      # What the participant is doing (e.g. bench, squat, etc.)
        category = f.split("-")[2].replace("_MetaWear_2019", "").rstrip("123")

        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df], ignore_index=True)

        elif "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df], ignore_index=True)  # Gyroscope data is sampled at 25Hz, so we will have more rows in this dataset.

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    # Drop columns
    del acc_df["epoch (ms)"], acc_df["time (01:00)"], acc_df["elapsed (s)"]
    del gyro_df["epoch (ms)"], gyro_df["time (01:00)"], gyro_df["elapsed (s)"]

    return acc_df, gyro_df


acc_df, gyro_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging dataframes
# --------------------------------------------------------------

df_merged = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)
df_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

# Define aggregation fuctions for resampling
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyro_x": "mean",
    "gyro_y": "mean",
    "gyro_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

# Resample: downsample (decreasing the frequency) the gyroscope data to match the accelerometer data
# This groups data per the new frequency and applies an aggregation function to the grouped data.
df_merged.resample(rule="200ms").apply(sampling).dropna()

# For computational efficiency, split the dataframe by day -> resample -> concatenate
groups = df_merged.groupby(pd.Grouper(freq="D"))
df_by_day = [group for key, group in groups]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in df_by_day])

# Change the datatype of set to integer
data_resampled.info()
data_resampled["set"] = data_resampled["set"].astype(int)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")

# %%
