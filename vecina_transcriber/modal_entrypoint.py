"""
This module serves as the entry point for the VECINA podcast transcription tool's modal interface.
Serves the cli.py functionality in a modal environment.

This module provides Modal deployment capabilities for the VECINA transcription tool,
allowing for scalable cloud-based audio transcription using Modal's serverless platform.
"""

import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import modal

from vecina_transcriber.transcriber import EnglishTranscriber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app definition
app = modal.App("vecina-transcriber")

# Define the Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("openai-whisper torch numpy")
    .add_local_dir("vecina_transcriber")
    .add_local_dir("_data")  # Include local module
    .pip_install_from_pyproject("pyproject.toml")
    .apt_install(["ffmpeg"])  # Required for audio processing
)

# Shared volume for model caching
volume = modal.Volume.from_name("whisper-models", create_if_missing=True)

# Model cache directory
MODEL_CACHE_DIR = "/cache/whisper"


@app.function(
    image=image,
    volumes={MODEL_CACHE_DIR: volume},
    gpu="T4",  # Use T4 GPU for faster processing
    timeout=3600,  # 1 hour timeout
    memory=8192,  # 8GB memory
)
def transcribe_audio_modal(
    audio_data: bytes,
    filename: str,
    model_name: str = "base",
    **transcription_kwargs: Any
) -> Dict[str, Any]:
    """
    Transcribe audio data using Whisper in Modal environment.

    Args:
        audio_data: Raw audio file bytes
        filename: Original filename for context
        model_name: Whisper model to use
        **transcription_kwargs: Additional transcription parameters

    Returns:
        Dictionary containing transcription results
    """
    logger.info(
        f"Starting transcription of {filename} using model {model_name}")

    try:
        # Create transcriber with cached model directory
        transcriber = EnglishTranscriber(
            model_name=model_name,
            download_root=MODEL_CACHE_DIR
        )

        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            temp_file.write(audio_data)
            temp_audio_path = temp_file.name

        # Transcribe the audio
        result = transcriber.transcribe(
            audio=temp_audio_path,
            verbose=True,
            **transcription_kwargs
        )

        # Clean up temporary file
        Path(temp_audio_path).unlink()

        # Add metadata
        result.update({
            "filename": filename,
            "model_used": model_name,
            "processed_in_modal": True
        })

        logger.info(f"Transcription completed for {filename}")
        return result

    except Exception as e:
        logger.error(f"Transcription failed for {filename}: {e}")
        raise


@app.function(
    image=image,
    volumes={MODEL_CACHE_DIR: volume},
    gpu="T4",
    timeout=7200,  # 2 hour timeout for batch processing
    memory=8192,
)
def batch_transcribe_modal(
    audio_files_data: List[Dict[str, Union[bytes, str]]],
    model_name: str = "base",
    **transcription_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Batch transcribe multiple audio files in Modal environment.

    Args:
        audio_files_data: List of dicts with 'data' (bytes) and 'filename' (str)
        model_name: Whisper model to use
        **transcription_kwargs: Additional transcription parameters

    Returns:
        List of transcription results
    """
    logger.info(
        f"Starting batch transcription of {len(audio_files_data)} files")

    results = []

    # Create transcriber once for all files
    transcriber = EnglishTranscriber(
        model_name=model_name,
        download_root=MODEL_CACHE_DIR
    )

    for file_data in audio_files_data:
        try:
            filename = file_data["filename"]
            audio_data = file_data["data"]

            logger.info(f"Processing {filename}")

            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(audio_data)
                temp_audio_path = temp_file.name

            # Transcribe the audio
            result = transcriber.transcribe(
                audio=temp_audio_path,
                verbose=False,  # Reduce verbosity for batch processing
                **transcription_kwargs
            )

            # Clean up temporary file
            Path(temp_audio_path).unlink()

            # Add metadata
            result.update({
                "filename": filename,
                "model_used": model_name,
                "processed_in_modal": True
            })

            results.append(result)
            logger.info(f"Completed {filename}")

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            # Add error result
            results.append({
                "filename": file_data["filename"],
                "error": str(e),
                "processed_in_modal": True
            })

    logger.info(f"Batch transcription completed. {len(results)} results")
    return results


@app.function(
    image=image,
    volumes={MODEL_CACHE_DIR: volume},
    timeout=300,  # 5 minute timeout
)
def get_model_info_modal(model_name: str = "base") -> Dict[str, Any]:
    """
    Get information about a Whisper model in Modal environment.

    Args:
        model_name: Name of the Whisper model

    Returns:
        Model information dictionary
    """
    logger.info(f"Getting info for model {model_name}")

    try:
        transcriber = EnglishTranscriber(
            model_name=model_name,
            download_root=MODEL_CACHE_DIR
        )

        model_info = transcriber.get_model_info()
        model_info["modal_environment"] = True

        return model_info

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise


@app.function(image=image, timeout=60)
def list_available_models_modal() -> List[str]:
    """
    List available Whisper models.

    Returns:
        List of available model names
    """
    return EnglishTranscriber.get_available_models()


# Convenience functions for external usage
def transcribe_with_modal(
    audio_file_path: Union[str, Path],
    model_name: str = "base",
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Convenience function to transcribe a local audio file using Modal.

    Args:
        audio_file_path: Path to local audio file
        model_name: Whisper model to use
        **kwargs: Additional transcription parameters

    Returns:
        Transcription result dictionary
    """
    audio_path = Path(audio_file_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Read audio file
    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    # Run transcription in Modal
    with app.run():
        result = transcribe_audio_modal.remote(
            audio_data=audio_data,
            filename=audio_path.name,
            model_name=model_name,
            **kwargs
        )

    return result


def batch_transcribe_with_modal(
    audio_file_paths: List[Union[str, Path]],
    model_name: str = "base",
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Convenience function to batch transcribe local audio files using Modal.

    Args:
        audio_file_paths: List of paths to local audio files
        model_name: Whisper model to use
        **kwargs: Additional transcription parameters

    Returns:
        List of transcription results
    """
    audio_files_data = []

    for file_path in audio_file_paths:
        audio_path = Path(file_path)
        if not audio_path.exists():
            logger.warning(f"Audio file not found, skipping: {audio_path}")
            continue

        with open(audio_path, 'rb') as f:
            audio_data = f.read()

        audio_files_data.append({
            "filename": audio_path.name,
            "data": audio_data
        })

    # Run batch transcription in Modal
    with app.run():
        results = batch_transcribe_modal.remote(
            audio_files_data=audio_files_data,
            model_name=model_name,
            **kwargs
        )

    return results


if __name__ == "__main__":
    # Example usage when running directly
    import sys

    if len(sys.argv) < 2:
        print("Usage: python modal_entrypoint.py <audio_file> [model_name]")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"

    print(f"Transcribing {audio_file} using model {model_name} in Modal...")

    try:
        result = transcribe_with_modal(audio_file, model_name)
        print("\nTranscription completed!")
        print(f"Text: {result['text'][:200]}...")  # Show first 200 chars
        print(f"Segments: {len(result['segments'])}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
