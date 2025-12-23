# -*- coding: utf-8 -*-
"""
Version améliorée pour limiter les "phrases oubliées" (sans changer de bibliothèque Whisper).

Ajouts principaux :
1) Paramètres Whisper plus tolérants (garde plus de parole faible)
2) Option "fallback" si Whisper hésite (temperature en liste)
3) Pré-traitement audio avec ffmpeg (normalisation du volume + réduction légère du bruit)
4) Sortie en 2 fichiers : brut segments + texte en phrases (pour diagnostiquer ce qui manque)
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

# --- NLTK (si besoin) ---
nltk.data.path.append("D:/CODING/.params/nltk_data")
# nltk.download('punkt', force=True)

# =========================
# VARIABLES
# =========================
nom_fichier_source = "D:/CODING/.entree/stiegler.mp4"

# fichiers temporaires
fichier_audio_extrait = "D:/CODING/.data/_tmp_extrait.wav"         # si vidéo
fichier_sortie_audio = "D:/CODING/.data/_tmp_16k_mono.wav"         # wav 16k mono
fichier_sortie_audio_clean = "D:/CODING/.data/_tmp_clean.wav"      # wav nettoyé (ffmpeg)
fichier_sortie_texte = "D:/CODING/.data/transcript.txt"            # sortie finale en phrases
fichier_sortie_segments = "D:/CODING/.data/transcript_segments.txt"  # sortie debug segments

# langues
langue_token = "french"   # pour nltk
langue_audio = "fr"       # pour whisper

# modèle whisper
WHISPER_MODEL = "medium" #tiny, base, small, medium, large-v3

# Extensions courantes
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

# =========================
# RÉGLAGES POUR "PHRASES OUBLIÉES"
# =========================
# Plus bas = Whisper garde davantage de passages faibles (mais risque un peu plus de faux texte dans les silences)
NO_SPEECH_THRESHOLD = 0.10   # au lieu de 0.60 (essaie 0.4 si tu veux plus prudent)

# Permet à Whisper de refaire un essai si ce qu'il sort est mauvais (utile avec bruit, accents, mots avalés)
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4) #TEMPERATURE_FALLBACK = (0.0) pour aller plus vite, TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4) pour être plus précis

# Un peu plus tolérant quand Whisper est "pas sûr"
LOGPROB_THRESHOLD = -2.0          # au lieu de -1.0
COMPRESSION_RATIO_THRESHOLD = 3.0  # au lieu de 2.4

# Beam search : plus lent mais souvent plus complet - mettre 1 pour aller plus vite, 5 pour etre plus précis
BEAM_SIZE = 5
BEST_OF = 5

# Pré-traitement ffmpeg (fortement recommandé si voix faible / bruit)
USE_FFMPEG_CLEAN = True
# loudnorm = normalise le volume (souvent le plus gros gain)
# afftdn = réduction de bruit légère (utile si souffle constant)
FFMPEG_FILTER = "loudnorm,afftdn"


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
    """
    Normalise le volume + réduit un peu le bruit.
    Objectif : éviter que Whisper rate des phrases faibles.
    """
    if not USE_FFMPEG_CLEAN:
        return in_wav

    if not ffmpeg_exists():
        log("[WARN] ffmpeg introuvable dans le PATH, nettoyage audio ignoré.")
        return in_wav

    log("Nettoyage audio via ffmpeg (volume + bruit)…")
    cmd = [
        "ffmpeg", "-y",
        "-i", in_wav,
        "-af", FFMPEG_FILTER,
        out_wav
    ]
    # capture la sortie sans spammer le terminal
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        log("[WARN] ffmpeg a échoué, nettoyage audio ignoré.")
        log(p.stderr[-500:])  # affiche la fin du log pour debug
        return in_wav

    return out_wav


def transcribe_with_whisper(wav_path: str, language: str) -> dict:
    log(f"Chargement du modèle Whisper: {WHISPER_MODEL} …")
    model = whisper.load_model(WHISPER_MODEL)

    log("Transcription en cours…")
    result = model.transcribe(
        wav_path,
        language=language,

        # Qualité / complétude :
        beam_size=BEAM_SIZE,
        best_of=BEST_OF,

        # Fallback : si 0.0 est mauvais sur un passage, Whisper réessaie avec 0.2 puis 0.4
        temperature=TEMPERATURE_FALLBACK,

        condition_on_previous_text=True,

        # IMPORTANT pour les phrases "oubliées" (voix faible / bruit / silences mal détectés)
        no_speech_threshold=NO_SPEECH_THRESHOLD,

        # Plus tolérant quand Whisper n'est pas très sûr
        logprob_threshold=LOGPROB_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,

        verbose=False
    )
    return result


def write_segments_debug(result: dict, out_txt_path: str) -> None:
    """
    Écrit chaque segment Whisper sur une ligne, avec timecodes.
    Très utile pour voir si Whisper a bien reconnu la phrase mais que le découpage final l'a "perdue".
    """
    segments = result.get("segments", [])
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = (seg.get("text") or "").strip()
            if text:
                f.write(f"[{start:0.2f} -> {end:0.2f}] {text}\n")


def write_transcript(result: dict, out_txt_path: str) -> None:
    """
    Sortie finale : texte en phrases.
    On construit d'abord depuis les segments (plus fiable), puis NLTK.
    """
    log("Mise au format du texte…")

    segments = result.get("segments", [])
    if segments:
        texte = " ".join([(seg.get("text") or "").strip() for seg in segments if (seg.get("text") or "").strip()]).strip()
    else:
        texte = (result.get("text") or "").strip()

    phrases = sent_tokenize(texte, language=langue_token) if texte else []

    log("Création du fichier de sortie…")
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for p in phrases:
            p = p.strip()
            if p:
                f.write(p + "\n")


def main():
    log("Début du programme")

    # nettoyage sorties/temporaires
    log("Suppression des fichiers temporaires/anciens résultats…")
    for p in [fichier_sortie_texte, fichier_sortie_segments, fichier_sortie_audio, fichier_sortie_audio_clean, fichier_audio_extrait]:
        safe_remove(p)

    # 1) audio depuis vidéo si besoin
    audio_path = extract_audio_if_video(nom_fichier_source, fichier_audio_extrait)

    # 2) conversion standard pour whisper
    convert_to_16k_mono_wav(audio_path, fichier_sortie_audio)

    # 3) nettoyage audio (optionnel mais recommandé)
    wav_for_whisper = clean_audio_with_ffmpeg(fichier_sortie_audio, fichier_sortie_audio_clean)

    # 4) transcription
    result = transcribe_with_whisper(wav_for_whisper, langue_audio)

    # 5) debug segments (pour vérifier ce qui manque vraiment)
    write_segments_debug(result, fichier_sortie_segments)

    # 6) sortie texte finale
    write_transcript(result, fichier_sortie_texte)

    # 7) nettoyage (NE JAMAIS supprimer le fichier source)
    log("Nettoyage des fichiers temporaires…")
    safe_remove(fichier_sortie_audio)
    safe_remove(fichier_sortie_audio_clean)
    if audio_path == fichier_audio_extrait:
        safe_remove(fichier_audio_extrait)

    log("Fin du programme")


if __name__ == "__main__":
    main()
