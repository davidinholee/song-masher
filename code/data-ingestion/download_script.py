#pip install moviepy
#pip install pytube

# Code to download a Youtube playlist and save each song as an mp3 locally
from pytube import YouTube
from pytube import Playlist
import moviepy.editor as mp
import re


playlist_url = 'https://www.youtube.com/watch?v=W81OL2X39ts&list=PLv3TTBr1W_9s-I8N3SvDld1QFbJsCJzzW&ab_channel=CagemanMashUps'
local_folder = ''

# Download all videos from playlist
playlist = Playlist(playlist_url)
for url in playlist:
    YouTube(url).streams.first().download()

# Retain only the downloaded audio file (mp3, not mp4)
for file in os.listdir(local_folder):
  if re.search('mp4', file):
    mp4_path = os.path.join(local_folder, file)
    mp3_path = os.path.join(local_folder, os.path.splitext(file)[0] + '.mp3')
    new_file = mp.AudioFileClip(mp4_path)
    new_file.write_audiofile(mp3_path)
    os.remove(mp4_path)