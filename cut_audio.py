import os
import re
from pydub import AudioSegment

def extract_time_stamps(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()
    timestamps = re.findall(r'(\d+)_(\d+)', contents)
    return timestamps

def split_audio(audio_path, timestamps, save_dir):
    audio = AudioSegment.from_wav(audio_path)
    time, _ = timestamps[0]
    for i, (start, end) in enumerate(timestamps):
        start_time = int(start) - int(time)
        end_time = int(end) - int(time)
        segment = audio[start_time:end_time]
        segment.export(os.path.join(save_dir, f'{i+1:03d}.wav'), format='wav')

input_dir = '../PhaseII_data'
output_dir_base = 'data/Clipped_Commands_PhaseII'

for session_dir in sorted(os.listdir(input_dir)):
    session_path = f"{input_dir}/{session_dir}"
    for task_dir in sorted(os.listdir(session_path)):
        task_path = f"{session_path}/{task_dir}"
        cha_files = [f for f in os.listdir(f"{task_path}") if f.endswith(".cha")]

        for idx, cha_file in enumerate(cha_files):
            cha_path = os.path.join(task_path, cha_file)
            wav_file = cha_file.replace('.cha', '.wav')
            wav_path = os.path.join(task_path, wav_file)
            if not os.path.isfile(wav_path):
                continue

            timestamps = extract_time_stamps(cha_path)

            cha_dir_name = os.path.splitext(cha_file)[0]
            output_dir = f"{output_dir_base}/{session_dir}/{task_dir}/{cha_dir_name}"
            os.makedirs(output_dir, exist_ok=True)

            split_audio(wav_path, timestamps, output_dir)

split_audio(wav_path, timestamps, output_dir)