#pip install moviepy
#pip install pytube
#pip install pydrive 

from pytube import YouTube
from pytube import Playlist
import os
import moviepy.editor as mp
import re
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import glob

#code to download the Youtube playlist and save each song as an mp3 in a folder 

playlist = Playlist("https://www.youtube.com/watch?v=sTKY5GTQ1HQ&list=PLe5mBZmNVoYqnWVoFA22e6Q95HNFUv9t5&ab_channel=SelinaStarr")
counter = 0
index_start = 880
song_num = index_start
print(playlist)
seen = 0
for url in playlist:
  if seen == 0:
    YouTube(url).streams.first().download('/Users/jamescai/Desktop/CS1470/song-masher/data/mashup-mp3-3', filename_prefix=str(song_num) + ' ' + seen + ' ')
    print(counter)
    seen = 1
  else:
    YouTube(url).streams.first().download('/Users/jamescai/Desktop/CS1470/song-masher/data/mashup-mp3-3', filename_prefix=str(song_num) + ' ' + seen + ' ')
    print(counter)
    seen = 0
    counter += 1
  song_num = index_start + counter

folder = "/Users/jamescai/Desktop/CS1470/song-masher/data/mashup-mp3-3"
for file in os.listdir(folder):
  if re.search('mp4', file):
    mp4_path = os.path.join(folder,file)
    mp3_path = os.path.join(folder,os.path.splitext(file)[0]+'.mp3')
    new_file = mp.AudioFileClip(mp4_path)
    new_file.write_audiofile(mp3_path)
    os.remove(mp4_path)


"""
#code to upload files to google drive

#handles google drive API authentication 
g_login = GoogleAuth()
g_login.LocalWebserverAuth()
drive = GoogleDrive(g_login)

# Upload each file in the mashups folder
folder = "/Users/nelso/Documents/SENIOR_FALL/final_project/mashups"
os.chdir("mashups")
for file in glob.glob("*.mp3"):
  with open(file, "r") as f:
    # Filename
    basename = os.path.basename(f.name)
    #filename = str(idx) + ' ' + str(basename)

    # Creates file
    new_file = drive.CreateFile({ 'title': basename })
    new_file.SetContentFile(str(basename))
    new_file.Upload() # Files.insert()
    print("File " + basename + " uploaded")
"""
