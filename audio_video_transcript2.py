# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import whisper


# ============================================================
# PARAMÈTRES À MODIFIER DIRECTEMENT ICI
# ============================================================

# Fichier source : audio ou vidéo
FICHIER_SOURCE = "C:/PYTHON/.entree/enregistrement.m4a"

# Dossier de sortie
DOSSIER_SORTIE = "C:/PYTHON/.data"

# Modèle Whisper : tiny, base, small, medium, large-v3
MODELE_WHISPER = "medium"

# Langue du fichier audio
LANGUE = "fr"

# True = applique un petit nettoyage audio
NETTOYAGE_AUDIO = True

# Supprimer le fichier WAV temporaire à la fin
SUPPRIMER_WAV_TEMP = True

# Filtre ffmpeg léger :
# - loudnorm : normalise le volume
# - afftdn   : réduit un peu le bruit
FILTRE_FFMPEG = "loudnorm,afftdn"


# ============================================================
# RÉGLAGES TECHNIQUES
# ============================================================

NO_SPEECH_THRESHOLD = 0.10
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4)
LOGPROB_THRESHOLD = -2.0
COMPRESSION_RATIO_THRESHOLD = 3.0
BEAM_SIZE = 5
BEST_OF = 5

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


# ============================================================
# OUTILS
# ============================================================

def log(message: str) -> None:
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {message}")


def erreur(message: str, code: int = 1) -> None:
    print(f"ERREUR - {message}", file=sys.stderr)
    sys.exit(code)


def verifier_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        erreur(
            "ffmpeg est introuvable. Installe ffmpeg puis vérifie qu'il est bien dans le PATH."
        )


def verifier_fichier_source(chemin: Path) -> None:
    if not chemin.exists():
        erreur(f"Le fichier source est introuvable : {chemin}")

    extension = chemin.suffix.lower()
    if extension not in AUDIO_EXTENSIONS and extension not in VIDEO_EXTENSIONS:
        erreur(
            f"Extension non prise en charge : {extension}\n"
            f"Formats audio acceptés : {sorted(AUDIO_EXTENSIONS)}\n"
            f"Formats vidéo acceptés : {sorted(VIDEO_EXTENSIONS)}"
        )


def creer_dossier(chemin: Path) -> None:
    chemin.mkdir(parents=True, exist_ok=True)


def supprimer_fichier_si_existe(chemin: Path) -> None:
    try:
        if chemin.exists():
            chemin.unlink()
    except Exception as e:
        log(f"[WARN] Impossible de supprimer {chemin} : {e}")


def executer_commande(cmd: list[str], message_erreur: str) -> None:
    resultat = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if resultat.returncode != 0:
        details = resultat.stderr[-2000:] if resultat.stderr else "Aucun détail."
        erreur(f"{message_erreur}\n\nDétail ffmpeg :\n{details}")


def convertir_en_wav_temporaire(
    fichier_source: Path,
    fichier_wav: Path,
    nettoyage_audio: bool,
    filtre_ffmpeg: str,
) -> None:
    log("Conversion du fichier source en WAV mono 16 kHz...")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(fichier_source),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
    ]

    if nettoyage_audio and filtre_ffmpeg.strip():
        cmd += ["-af", filtre_ffmpeg]

    cmd.append(str(fichier_wav))

    executer_commande(cmd, "Échec de la conversion vers WAV.")


def charger_modele_whisper(nom_modele: str):
    log(f"Chargement du modèle Whisper : {nom_modele}")
    try:
        return whisper.load_model(nom_modele)
    except Exception as e:
        erreur(f"Impossible de charger le modèle Whisper '{nom_modele}' : {e}")


def transcrire_audio(modele, fichier_wav: Path, langue: str) -> dict:
    log("Transcription en cours...")

    try:
        resultat = modele.transcribe(
            str(fichier_wav),
            language=langue,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            temperature=TEMPERATURE_FALLBACK,
            condition_on_previous_text=True,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            logprob_threshold=LOGPROB_THRESHOLD,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            verbose=False,
            fp16=False,
        )
        return resultat
    except Exception as e:
        erreur(f"Erreur pendant la transcription Whisper : {e}")


def ecrire_texte_simple(resultat: dict, fichier_sortie: Path) -> None:
    texte = (resultat.get("text") or "").strip()
    with fichier_sortie.open("w", encoding="utf-8") as f:
        if texte:
            f.write(texte + "\n")


def ecrire_segments(resultat: dict, fichier_sortie: Path) -> None:
    segments = resultat.get("segments", [])

    with fichier_sortie.open("w", encoding="utf-8") as f:
        for segment in segments:
            debut = segment.get("start", 0.0)
            fin = segment.get("end", 0.0)
            texte = (segment.get("text") or "").strip()

            if texte:
                f.write(f"[{debut:0.2f} -> {fin:0.2f}] {texte}\n")


def ecrire_json(resultat: dict, fichier_sortie: Path) -> None:
    with fichier_sortie.open("w", encoding="utf-8") as f:
        json.dump(resultat, f, ensure_ascii=False, indent=2)


def construire_noms_fichiers(dossier_sortie: Path, fichier_source: Path) -> dict[str, Path]:
    nom_base = fichier_source.stem

    return {
        "wav_temp": dossier_sortie / f"{nom_base}_temp_16k.wav",
        "txt": dossier_sortie / f"{nom_base}_transcription.txt",
        "segments": dossier_sortie / f"{nom_base}_segments.txt",
        "json": dossier_sortie / f"{nom_base}_transcription.json",
    }


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    fichier_source = Path(FICHIER_SOURCE)
    dossier_sortie = Path(DOSSIER_SORTIE)

    log("Début du programme")

    verifier_fichier_source(fichier_source)
    verifier_ffmpeg()
    creer_dossier(dossier_sortie)

    fichiers = construire_noms_fichiers(dossier_sortie, fichier_source)

    # Nettoyage des anciens fichiers de même nom
    for chemin in fichiers.values():
        supprimer_fichier_si_existe(chemin)

    # 1) Conversion en WAV
    convertir_en_wav_temporaire(
        fichier_source=fichier_source,
        fichier_wav=fichiers["wav_temp"],
        nettoyage_audio=NETTOYAGE_AUDIO,
        filtre_ffmpeg=FILTRE_FFMPEG,
    )

    # 2) Chargement du modèle
    modele = charger_modele_whisper(MODELE_WHISPER)

    # 3) Transcription
    resultat = transcrire_audio(
        modele=modele,
        fichier_wav=fichiers["wav_temp"],
        langue=LANGUE,
    )

    # 4) Sauvegarde
    ecrire_texte_simple(resultat, fichiers["txt"])
    ecrire_segments(resultat, fichiers["segments"])
    ecrire_json(resultat, fichiers["json"])

    # 5) Nettoyage
    if SUPPRIMER_WAV_TEMP:
        supprimer_fichier_si_existe(fichiers["wav_temp"])

    log("Transcription terminée")
    log(f"Fichier texte     : {fichiers['txt']}")
    log(f"Fichier segments  : {fichiers['segments']}")
    log(f"Fichier JSON      : {fichiers['json']}")

    if not SUPPRIMER_WAV_TEMP:
        log(f"Fichier WAV temp  : {fichiers['wav_temp']}")


if __name__ == "__main__":
    main()