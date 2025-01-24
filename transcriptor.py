import os
import torch
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm  # Pour les barres de progression
from typing import List, Dict
import warnings

# Détection du matériel
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
COMPUTE_TYPE = "float16" if USE_CUDA else "int8"

# Configuration des modèles
HF_TOKEN = ""  # Remplacez par votre token HuggingFace
MODEL_SIZE = "base"  # Peut être changé en small/medium/large
SEGMENT_DURATION = 30  # Durée des segments en secondes (ajustez si nécessaire)


def segment_audio(audio_path: str, segment_duration: int, output_dir: str) -> List[str]:
    """Divise l'audio en segments plus petits et les sauvegarde dans un dossier."""
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio) / 1000  # Durée totale en secondes
    segments = []

    # Créer le dossier pour les segments
    os.makedirs(output_dir, exist_ok=True)

    print(f"Durée totale de l'audio : {duration:.2f} secondes.")
    for i in range(0, int(duration), segment_duration):
        start = i * 1000  # Début en millisecondes
        end = min((i + segment_duration) * 1000, len(audio))  # Fin en millisecondes
        segment = audio[start:end]
        segment_path = os.path.join(output_dir, f"{Path(audio_path).stem}_segment_{i // segment_duration}.wav")
        segment.export(segment_path, format="wav")
        segments.append(segment_path)

    print(f"Audio découpé en {len(segments)} segments.")
    return segments


def process_diarization(audio_segments: List[str], diarization_pipeline) -> List[Dict]:
    """Applique la diarisation sur chaque segment."""
    diarization_results = []

    with tqdm(total=len(audio_segments), desc="Diarisation des segments") as pbar:
        for segment_path in audio_segments:
            diarization = diarization_pipeline(segment_path)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diarization_results.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "segment": segment_path
                })
            pbar.update(1)

    return diarization_results


def process_transcription(diarization_results: List[Dict], whisper_model) -> List[Dict]:
    """Applique la transcription sur chaque segment et synchronise avec les locuteurs."""
    transcription_results = []

    with tqdm(total=len(diarization_results), desc="Transcription des segments") as pbar:
        for result in diarization_results:
            segment_path = result["segment"]

            # Transcrire le segment
            segments, _ = whisper_model.transcribe(segment_path, language="fr", beam_size=5)

            # Associer les résultats de transcription au locuteur
            for segment in segments:
                transcription_results.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": result["speaker"],
                    "text": segment.text,
                    "segment": segment_path
                })
            pbar.update(1)

    return transcription_results


def save_transcriptions_to_file(transcription_results: List[Dict], output_file: str):
    """Enregistre les transcriptions dans un fichier texte."""
    speaker_mapping = {}  # Pour mapper SPEAKER_00, SPEAKER_01 à Speaker 1, Speaker 2
    speaker_counter = 1

    with open(output_file, "w", encoding="utf-8") as file:
        for result in transcription_results:
            speaker_id = result["speaker"]

            # Créer un mapping unique pour chaque locuteur
            if speaker_id not in speaker_mapping:
                speaker_mapping[speaker_id] = f"Speaker {speaker_counter}"
                speaker_counter += 1

            speaker_name = speaker_mapping[speaker_id]
            file.write(f"{speaker_name}: {result['text']}\n\n")

    print(f"Transcriptions sauvegardées dans {output_file}")


def main(audio_path: str):
    # Vérifiez que le fichier existe
    if not os.path.exists(audio_path):
        print(f"Erreur : Le fichier {audio_path} n'existe pas.")
        return

    # Dossier pour les segments
    segments_dir = os.path.join(Path(audio_path).parent, "segments")

    # Découper l'audio en segments
    print("Découpage de l'audio en segments...")
    audio_segments = segment_audio(audio_path, SEGMENT_DURATION, segments_dir)
    print(f"{len(audio_segments)} segments créés dans le dossier '{segments_dir}'.")

    # Charger la pipeline de diarisation
    print("Chargement de la pipeline de diarisation...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    ).to(torch.device(DEVICE))

    # Appliquer la diarisation
    print("Diarisation en cours...")
    diarization_results = process_diarization(audio_segments, diarization_pipeline)

    # Charger le modèle Whisper
    print("Chargement du modèle Whisper...")
    whisper_model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    # Appliquer la transcription
    print("Transcription en cours...")
    transcription_results = process_transcription(diarization_results, whisper_model)

    # Enregistrer les transcriptions dans un fichier texte
    output_file = os.path.join(Path(audio_path).parent, "transcriptions.txt")
    save_transcriptions_to_file(transcription_results, output_file)

    # Supprimer les segments temporaires
    print("\nNettoyage des segments temporaires...")
    for segment_path in audio_segments:
        os.remove(segment_path)
    os.rmdir(segments_dir)  # Supprime le dossier s'il est vide
    print("Segments temporaires supprimés.")


if __name__ == "__main__":
    # Ignorer les avertissements non critiques
    warnings.filterwarnings("ignore", message="std(): degrees of freedom is <= 0")

    # Entrer le chemin de l'audio
    audio_file = input("Chemin du fichier audio : ").strip()
    main(audio_file)