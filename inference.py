from __future__ import unicode_literals, division, print_function
from visualization import visualize_single
from helper_functions import class_names_from_csv, ensure_sample_rate

import tensorflow as tf
import tensorflow_hub as hub
from IPython.display import Audio
from scipy.io import wavfile
import pandas as pd
import os


def inference(df):
    columns = ['video_id', 'start_seconds', 'end_seconds', 'positive_labels', 'scores', 'labels']
    finished_df = pd.DataFrame(columns=columns)

    model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    for i, row in df.iterrows():
        video_id = row["YTID,"][:-1]
        clean_id = video_id.replace("-", "")
        wav_file_name = 'audio/' + clean_id + '.wav'
        if not os.path.isfile(wav_file_name):
            continue

        sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

        # Show some basic information about the audio.
        duration = len(wav_data) / sample_rate

        Audio(wav_data, rate=sample_rate)
        waveform = wav_data / tf.int16.max
        scores, embeddings, spectrogram = model(waveform)

        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()
        inferred_class = class_names[scores_np.mean(axis=0).argmax()]
        print(f'The main sound is: {inferred_class}')

        x, y = visualize_single(waveform, spectrogram_np, scores, scores_np, class_names)
        print(f"File {wav_file_name} is done!")

        temp = [row['YTID,'], row['start_seconds,'], row['end_seconds,'], row['positive_labels'], x, y]
        temp_df = pd.DataFrame([temp], columns=columns)
        finished_df = pd.concat([finished_df, temp_df])
        finished_df.to_csv('data.csv')
