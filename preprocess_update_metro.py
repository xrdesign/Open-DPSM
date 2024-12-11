import pandas as pd
import json
import os
import numpy as np
import cv2
import subprocess
from io import StringIO

from concurrent.futures import ThreadPoolExecutor

data_folder = "./Space Transit w- EEG/Test 11-04-2024 - 2nd test with EEG"
video_file = "./Space Transit w- EEG/Test 11-04-2024 - 2nd test with EEG/test_9998_obs_game2.mkv"

# Position Offsets for occlusion view window in video:
x_min, x_max = 73, 1846
y_min, y_max = 42, 1039
screen_width = int((x_max - x_min) / 2.0)
screen_height = int((y_max - y_min))
x_mid = x_min + screen_width

# Input filenames
csv_file = os.path.join(data_folder, 'test_9998_EyetrackingScreenPosition_game2.csv')
game_replay_file = os.path.join(data_folder, 'test_9998_game2.replay')

# Output filenames
# create a result folder in the data_folder
result_folder = os.path.join(data_folder, 'result')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
left_file, right_file = 'left_coordinates.csv', 'right_coordinates.csv'
left_output_path, right_output_path = 'left.mp4', 'right.mp4'

# join the result folder with the left and right file
left_file = os.path.join(result_folder, left_file)
right_file = os.path.join(result_folder, right_file)
left_output_path = os.path.join(result_folder, left_output_path)
right_output_path = os.path.join(result_folder, right_output_path)

# Sync event extraction from game replay file
print("Preprocessing input csv")
print("\tFinding sync time in replay file: ")
sync_time_logs = -1

with open(game_replay_file, 'r') as file:
    for line in file:
        data = json.loads(line)
        if data.get('EVENT_TYPE') == 'SyncGamesEvent':
            sync_time_logs = data['TIME']
            break

if sync_time_logs == -1:
    print("ERROR: No sync event found in replay file...")
    exit()

print(f"\tFound replay file sync time: {sync_time_logs}")

# Read and process input CSV
print("	Reading input csv")
data = pd.read_csv(csv_file)

print("\tTransforming normalized coordinates")
data['leftX'] = data['leftX'] * screen_width
data['leftY'] = (1 - data['leftY']) * screen_height
data['rightX'] = data['rightX'] * screen_width
data['rightY'] = (1 - data['rightY']) * screen_height

# clamp to between 0 and screen_width/height
data['leftX'] = np.clip(data['leftX'], 0, screen_width)
data['leftY'] = np.clip(data['leftY'], 0, screen_height)
data['rightX'] = np.clip(data['rightX'], 0, screen_width)
data['rightY'] = np.clip(data['rightY'], 0, screen_height)

data['timestamp'] -= sync_time_logs

print("\tFiltering negative timestamps")
filtered_data = data[data['timestamp'] >= 0].dropna()
end_time_eyetracking = filtered_data['timestamp'].iloc[-1]

# Video processing to find sync time
print("Opening input video to find sync time...")
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sync_time_video = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    if sync_time_video == -1:
        pixel_color = frame[y_max - 20, x_mid]
        b, g, r = pixel_color
        if current_time > 1 and not (b == 255 and r == 255 and g == 255):
            sync_time_video = current_time
            print(f"Video sync time: {sync_time_video}")
            end_time_video = frame_count / fps - sync_time_video

            print(f"\tEnd timestamp eyetracking = {end_time_eyetracking}")
            print(f"\tEnd timestamp video = {end_time_video}")
            end_time = min(end_time_eyetracking, end_time_video)
            print("Sync time found, stopping video read.")
            break

cap.release()

# Use FFmpeg to crop the video for left and right views, starting from sync time
print("Starting FFmpeg cropping for left and right views...")
left_ffmpeg_cmd = [
    'ffmpeg', '-y', '-i', video_file,
    '-ss', str(sync_time_video),
    '-t', str(end_time),
    '-vf', f'crop={screen_width}:{screen_height}:{x_min}:{y_min}',
    '-c:v', 'libx264', '-preset', 'ultrafast',
    left_output_path
]

right_ffmpeg_cmd = [
    'ffmpeg', '-y', '-i', video_file,
    '-ss', str(sync_time_video),
    '-t', str(end_time),
    '-vf', f'crop={screen_width}:{screen_height}:{x_mid}:{y_min}',
    '-c:v', 'libx264', '-preset', 'ultrafast',
    right_output_path
]

# # Run the FFmpeg commands
subprocess.run(left_ffmpeg_cmd)
subprocess.run(right_ffmpeg_cmd)

print("Video cropping completed.")

# Writing output CSVs
print("Writing output CSVs")
left_columns = ['timestamp', 'leftX', 'leftY', 'leftPupilDiameter']
right_columns = ['timestamp', 'rightX', 'rightY', 'rightPupilDiameter']
# append index column to the filtered data
# filtered_data['index'] = np.arange(len(filtered_data))
filtered_data[left_columns].to_csv(left_file, index=True)
filtered_data[right_columns].to_csv(right_file, index=True)
