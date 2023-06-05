from __future__ import unicode_literals
import os
import youtube_dl
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import librosa
import soundfile as sf


def download_files(df):
    for i, row in df.iterrows():
        try:
            video_id = row["YTID,"][:-1]

            # Start and end timestamps in seconds
            start_time = int(row["start_seconds,"][:-1].split(".")[0])
            end_time = int(row["end_seconds,"][:-1].split(".")[0])
            video_url = f'https://www.youtube.com/watch?v={video_id}'
            file_name = "long_" + video_id.replace("-", "") + ".wav"
            file_name_clipped = video_id.replace("-", "") + ".wav"

            ydl_opts = {
                'outtmpl': file_name,
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'postprocessor_args': ['-ar', '16000'],
                'prefer_ffmpeg': True,
                'keepvideo': False
            }

            download_wav(ydl_opts, video_url)
            clip_wav(file_name, file_name_clipped, start_time, end_time)
            
        except Exception as e:
            print(e)


def download_wav(ydl_opts, video_url):
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def clip_wav(input_name, output_name, start_time, end_time):
    x, _ = librosa.load(input_name, sr=16000)
    sf.write('tmp.wav', x, 16000)
    ffmpeg_extract_subclip('tmp.wav', start_time, end_time, targetname=output_name)
    os.remove(input_name)
    os.remove('tmp.wav')
