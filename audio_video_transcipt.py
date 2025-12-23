#bibliothèques
import os
from datetime import datetime
from colorama import Fore, Back, Style
from moviepy import VideoFileClip
from pydub import AudioSegment
import whisper

import nltk
from nltk.tokenize import sent_tokenize
nltk.data.path.append("C:/DATA/code/.params/nltk_data")
# nltk.download('punkt', force=True) a recharger si necessaire
# nltk.download('punkt_tab', force=True) a recharger si necessaire

#variables
nom_fichier_source = "C:/DATA/code/.data/eolia.m4a"

fichier_travail = "C:/DATA/code/.data/travail.wav"
fichier_sortie_audio = "C:/DATA/code/.data/sortie.wav"
fichier_sortie_texte = "C:/DATA/code/.data/transcript_brut.txt"
langue_token = "french"
langue_audio = "fr"

# Extensions courantes
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

#lancement
print("{} - Début du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

#suppression du fichier de sortie et du fichier de travail
print("{} - Suppression des fichiers de sortie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
if os.path.exists(fichier_sortie_texte):
    os.remove(fichier_sortie_texte)

if os.path.exists(fichier_sortie_audio):
    os.remove(fichier_sortie_audio)

#test si audio ou vidéo
print("{} - Test si fichier audio ou vidéo".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
ext = os.path.splitext(nom_fichier_source)[1].lower()

#Extraction de l'audio du fichier vidéo    
if ext in VIDEO_EXTENSIONS:
    print("{} - Fichier vidéo, extraction de l audio".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    video = VideoFileClip(nom_fichier_source) 
    audio_file = video.audio 
    audio_file.write_audiofile(fichier_travail)
else:
    fichier_travail = nom_fichier_source

# Conversion de l audio en mono 16
print("{} - conversion de l audio en mono 16".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
audio = AudioSegment.from_file(fichier_travail)
audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
audio.export(fichier_sortie_audio, format="wav")

# Extraction du texte
print("{} - Extraction du texte".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
model = whisper.load_model("medium")
result = model.transcribe(fichier_sortie_audio, language=langue_audio)

# Découper le texte en phrases avec nltk
print("{} - Mise au format du texte".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
phrases = sent_tokenize(result["text"], language=langue_token)

print("{} - Création du fichier de sortie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
with open(fichier_sortie_texte, 'w', encoding="utf-8") as fs:
    for phrase in phrases:
        fs.write(phrase + "\n")  # Écrit la phrase suivie d'un saut de ligne
    fs.close()

#suppression du fichier de travail
os.remove(fichier_travail)

print("{} - Fin du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))