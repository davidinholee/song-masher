from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import glob, os

# Login and create Drive obj
g_auth = GoogleAuth()
g_auth.LocalWebserverAuth()
drive = GoogleDrive(g_auth)

# Upload each file in the mashups folder
os.chdir("/mashups") # TODO
idx = 1
for file in glob.glob("*.mp3"):
  with open(file, "r") as f:
    # Filename
    basename = os.path.basename(f.name)
    filename = str(idx) + ' ' + str(basename)

    # Creates file
    new_file = drive.CreateFile({ 'title': str(filename) })
    new_file.SetContentFile(str(basename))
    new_file.Upload() # Files.insert()
    idx +=1

    print("File " + filename + " uploaded")