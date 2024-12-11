####
import pandas as pd
import cv2
import subprocess

train_size = 0.2 # 20% of the data will be used for training

path = './Space Transit w- EEG/Test 11-04-2024 - 2nd test with EEG/result/'
eye_data_path = 'right_coordinates.csv'
vid_data_path = 'right.mp4'

## output by default will be {eye_data_path}_train/val.csv and {vid_data_path}_train/val.mp4
# 
# Load the data
eye_data = pd.read_csv(path + eye_data_path)

# calculate the first train_size% seconds
train_time = eye_data['timestamp'].max() * train_size

# split the data
train_data = eye_data[eye_data['timestamp'] < train_time]
val_data = eye_data[eye_data['timestamp'] >= train_time]

# remove the first column because it's the index
train_data = train_data.iloc[:, 1:]
val_data = val_data.iloc[:, 1:]

# offset the timestamp for val_data so that it starts from 0
val_data['timestamp'] -= val_data['timestamp'].min()

# save the data
train_data.to_csv(path + eye_data_path[:-4] + '_train.csv', index=True)
val_data.to_csv(path + eye_data_path[:-4] + '_val.csv', index=True)

# split the video
# ffmpeg directly splits the video into train and val based on the video duration

# Open the video file
video = cv2.VideoCapture(path + vid_data_path)

# Check if the video was opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
else:
    # Get the total number of frames
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # Get the frames per second (fps)
    fps = video.get(cv2.CAP_PROP_FPS)
    # Calculate the duration in seconds
    video_duration = frame_count / fps
    print(f"Video duration: {video_duration} seconds")

# Release the video capture object
video.release()
train_duration = video_duration * train_size

print(f"Video duration: {video_duration}, train duration: {train_duration}")

# train from 0 ~ train_duration
train_ffmpeg_cmd = [
    'ffmpeg', '-y', '-i', path + vid_data_path,
    '-t', str(train_duration),
    path + vid_data_path[:-4] + '_train.mp4'
]

# val from train_duration ~ end
val_ffmpeg_cmd = [
    'ffmpeg', '-y', '-i', path + vid_data_path,
    '-ss', str(train_duration),
    path + vid_data_path[:-4] + '_val.mp4'
]

# Run the FFmpeg commands
subprocess.run(train_ffmpeg_cmd)
subprocess.run(val_ffmpeg_cmd)