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

playlist = Playlist("https://www.youtube.com/watch?v=wevyuNG_9Jo&list=UU_sMbyQqtLPZ9WjPcvOpbOA&index=22&ab_channel=oneboredjeuMashup")
counter = 1
for url in playlist:
    if counter == 137 or counter == 152 or counter ==226 or counter == 326 or counter == 336 or counter == 344 or counter == 352:
      print(counter)
      counter += 1
      continue
    else:
      song_num = counter - 2
      YouTube(url).streams.first().download('/Users/nelso/Documents/SENIOR_FALL/final_project/mashups', filename_prefix=str(song_num) + ' ')
      print(counter)
      counter += 1

folder = "/Users/nelso/Documents/SENIOR_FALL/final_project/mashups"
for file in os.listdir(folder):
  if re.search('mp4', file):
    mp4_path = os.path.join(folder,file)
    mp3_path = os.path.join(folder,os.path.splitext(file)[0]+'.mp3')
    new_file = mp.AudioFileClip(mp4_path)
    new_file.write_audiofile(mp3_path)
    os.remove(mp4_path)



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

