from os import path
from pydub import AudioSegment

# files
src = "/Users/jamescai/Desktop/CS1470/song-masher/data/mashup-mp3-3/933 Lose You To Adore You  Selena Gomez & Harry Styles Mashup!.mp3"
dst = "/Users/jamescai/Desktop/CS1470/song-masher/data/mashup-wav/933 Lose You To Adore You  Selena Gomez & Harry Styles Mashup!.wav"

# convert mp3 to wav
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")