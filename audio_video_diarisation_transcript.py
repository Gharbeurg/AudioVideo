# -*- coding: utf-8 -*-
"""
Ajout de la diarisation (séparation des locuteurs) avec pyannote.audio
- On garde Whisper pour la transcription
- On ajoute pyannote.audio pour détecter "qui parle quand"
- On associe chaque segment Whisper à un locuteur via l’horodatage

IMPORTANT :
- pyannote.audio nécessite en général un token Hugging Face + accepter les conditions du modèle.
- Mets ton token dans une variable d’environnement HF_TOKEN, ou directement dans HF_TOKEN ci-dessous.
"""

import os
import shutil
import subprocess
from datetime import datetime

from moviepy import VideoFileClip
from pydub import AudioSegment
import whisper

import nltk
from nltk.tokenize import sent_tokenize

# --- diarisation ---
# pip install pyannote.audio
from pyannote.audio import Pipeline

# --- NLTK (si besoin) ---
nltk.data.path.append("C:/PYTHON/.params/nltk_data")
# nltk.download('punkt', force=True)

# =========================
# VARIABLES
# =========================
nom_fichier_source = "C:/PYTHON/.entree/stiegler.mp4"

# fichiers temporaires
fichier_audio_extrait = "C:/PYTHON/.data/_tmp_extrait.wav"           # si vidéo
fichier_sortie_audio = "C:/PYTHON/.data/_tmp_16k_mono.wav"           # wav 16k mono
fichier_sortie_audio_clean = "C:/PYTHON/.data/_tmp_clean.wav"        # wav nettoyé (ffmpeg)

# sorties
fichier_sortie_texte = "C:/PYTHON/.data/transcript.txt"
fichier_sortie_segments = "C:/PYTHON/.data/transcript_segments.txt"
fichier_sortie_locuteurs = "C:/PYTHON/.data/transcript_speakers.txt"   # NOUVEAU

# langues
langue_token = "french"
langue_audio = "fr"

# modèle whisper
WHISPER_MODEL = "large-v3"

# Extensions courantes
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

# =========================
# RÉGLAGES
# =========================
NO_SPEECH_THRESHOLD = 0.60
TEMPERATURE_FALLBACK = 0.0
LOGPROB_THRESHOLD = -2.0
COMPRESSION_RATIO_THRESHOLD = 3.0
BEAM_SIZE = 2
BEST_OF = 2

USE_FFMPEG_CLEAN = True
FFMPEG_FILTER = "loudnorm,afftdn"

# =========================
# DIARISATION (pyannote)
# =========================
USE_DIARIZATION = True

# Mets ton token HF ici OU dans la variable d'environnement HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Modèle conseillé (récent). Peut demander d'accepter des conditions sur Hugging Face.
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {msg}")


def safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        log(f"[WARN] Impossible de supprimer {path}: {e}")


def is_video(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def extract_audio_if_video(src_path: str, out_wav_path: str) -> str:
    if not is_video(src_path):
        return src_path

    log("Fichier vidéo détecté, extraction de l'audio…")
    clip = VideoFileClip(src_path)
    try:
        # MoviePy récent : pas de verbose=...
        clip.audio.write_audiofile(out_wav_path, logger=None)
    finally:
        try:
            clip.close()
        except:
            pass

    return out_wav_path


def convert_to_16k_mono_wav(src_audio_path: str, out_wav_path: str) -> None:
    log("Conversion audio en mono 16kHz 16-bit…")
    audio = AudioSegment.from_file(src_audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(out_wav_path, format="wav")


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def clean_audio_with_ffmpeg(in_wav: str, out_wav: str) -> str:
    if not USE_FFMPEG_CLEAN:
        return in_wav
    if not ffmpeg_exists():
        log("[WARN] ffmpeg introuvable dans le PATH, nettoyage audio ignoré.")
        return in_wav

    log("Nettoyage audio via ffmpeg (volume + bruit)…")
    cmd = ["ffmpeg", "-y", "-i", in_wav, "-af", FFMPEG_FILTER, out_wav]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        log("[WARN] ffmpeg a échoué, nettoyage audio ignoré.")
        log(p.stderr[-500:])
        return in_wav
    return out_wav


def transcribe_with_whisper(wav_path: str, language: str) -> dict:
    log(f"Chargement du modèle Whisper: {WHISPER_MODEL} …")
    model = whisper.load_model(WHISPER_MODEL)

    log("Transcription en cours…")
    result = model.transcribe(
        wav_path,
        language=language,
        beam_size=BEAM_SIZE,
        best_of=BEST_OF,
        temperature=TEMPERATURE_FALLBACK,
        condition_on_previous_text=True,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        logprob_threshold=LOGPROB_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        verbose=False
    )
    return result


def write_segments_debug(result: dict, out_txt_path: str) -> None:
    segments = result.get("segments", [])
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = (seg.get("text") or "").strip()
            if text:
                f.write(f"[{start:0.2f} -> {end:0.2f}] {text}\n")


def write_transcript(result: dict, out_txt_path: str) -> None:
    log("Mise au format du texte…")
    segments = result.get("segments", [])
    if segments:
        texte = " ".join(
            [(seg.get("text") or "").strip() for seg in segments if (seg.get("text") or "").strip()]
        ).strip()
    else:
        texte = (result.get("text") or "").strip()

    phrases = sent_tokenize(texte, language=langue_token) if texte else []

    log("Création du fichier de sortie…")
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for p in phrases:
            p = p.strip()
            if p:
                f.write(p + "\n")


def diarize_audio(wav_path: str):
    """
    Retourne un objet diarisation pyannote qui contient les segments + labels (SPEAKER_00, etc.)
    """
    if not USE_DIARIZATION:
        return None

    if not HF_TOKEN:
        log("[WARN] HF_TOKEN manquant : diarisation ignorée. (mets HF_TOKEN dans tes variables d'environnement)")
        return None

    log("Diarisation (détection des locuteurs)…")
    pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
    diarization = pipeline(wav_path)
    return diarization


def speaker_at_time(diarization, t: float) -> str:
    """
    Renvoie le locuteur actif au temps t (en secondes). Sinon 'UNKNOWN'.
    """
    if diarization is None:
        return "UNKNOWN"

    # diarization.itertracks(yield_label=True) donne (segment, track, label)
    for segment, _, label in diarization.itertracks(yield_label=True):
        if segment.start <= t <= segment.end:
            return label
    return "UNKNOWN"


def write_transcript_with_speakers(result: dict, diarization, out_txt_path: str) -> None:
    """
    Écrit un fichier où chaque segment Whisper est précédé du locuteur estimé.
    """
    segments = result.get("segments", [])
    if not segments:
        safe_remove(out_txt_path)
        return

    log("Création de la sortie avec locuteurs…")
    with open(out_txt_path, "w", encoding="utf-8") as f:
        last_speaker = None
        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = (seg.get("text") or "").strip()
            if not text:
                continue

            # on prend le milieu du segment Whisper pour décider "qui parle"
            mid = (start + end) / 2.0
            spk = speaker_at_time(diarization, mid)

            # petit confort de lecture : on saute une ligne quand on change de locuteur
            if last_speaker is not None and spk != last_speaker:
                f.write("\n")
            last_speaker = spk

            f.write(f"{spk} [{start:0.2f}-{end:0.2f}] {text}\n")


def main():
    log("Début du programme")

    # nettoyage sorties/temporaires
    log("Suppression des fichiers temporaires/anciens résultats…")
    for p in [
        fichier_sortie_texte, fichier_sortie_segments, fichier_sortie_locuteurs,
        fichier_sortie_audio, fichier_sortie_audio_clean, fichier_audio_extrait
    ]:
        safe_remove(p)

    # 1) audio depuis vidéo si besoin
    audio_path = extract_audio_if_video(nom_fichier_source, fichier_audio_extrait)

    # 2) conversion standard pour whisper
    convert_to_16k_mono_wav(audio_path, fichier_sortie_audio)

    # 3) nettoyage audio (optionnel)
    wav_for_processing = clean_audio_with_ffmpeg(fichier_sortie_audio, fichier_sortie_audio_clean)

    # 4) diarisation (avant ou après whisper, ça marche ; ici on le fait avant pour gagner du temps si erreur de token)
    diarization = diarize_audio(wav_for_processing)

    # 5) transcription whisper
    result = transcribe_with_whisper(wav_for_processing, langue_audio)

    # 6) debug segments
    write_segments_debug(result, fichier_sortie_segments)

    # 7) sortie texte finale (sans locuteurs)
    write_transcript(result, fichier_sortie_texte)

    # 8) sortie avec locuteurs
    write_transcript_with_speakers(result, diarization, fichier_sortie_locuteurs)

    # 9) nettoyage (NE JAMAIS supprimer le fichier source)
    log("Nettoyage des fichiers temporaires…")
    safe_remove(fichier_sortie_audio)
    safe_remove(fichier_sortie_audio_clean)
    if audio_path == fichier_audio_extrait:
        safe_remove(fichier_audio_extrait)

    log("Fin du programme")


if __name__ == "__main__":
    main()
