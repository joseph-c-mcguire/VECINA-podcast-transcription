"""
This module serves as the entry point for the VECINA podcast transcription tool's modal interface.
Serves the cli.py functionality in a modal environment.

This module provides Modal deployment capabilities for the VECINA transcription tool,
allowing for scalable cloud-based audio transcription using Modal's serverless platform.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import modal

# NOTE: We avoid importing the transcriber (and thus whisper) at module import time
# so that `modal serve` works even if local env lacks the heavy whisper dependency.
# Imports are performed lazily inside functions that run in the Modal container.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app definition
app = modal.App("vecina-transcriber")

# Shared volumes: one for model cache, one for data (audio + transcripts)
model_volume = modal.Volume.from_name("whisper-models", create_if_missing=True)
data_volume = modal.Volume.from_name("vecina-data", create_if_missing=True)

# Define the Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    # FastAPI for ASGI app + project deps
    .pip_install("fastapi[standard]")
    .pip_install_from_pyproject("pyproject.toml")
    .apt_install(["ffmpeg"])  # Required for audio processing
    # Copy local package into container and install
    .add_local_dir("vecina_transcriber", remote_path="/root/vecina_transcriber", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .workdir("/root")

)

MODEL_CACHE_DIR = "/cache/whisper"
DATA_ROOT = "/data"
AUDIO_DIR = f"{DATA_ROOT}/audio"
TRANSCRIPTS_DIR = f"{DATA_ROOT}/transcripts"
CONFIG_PATH = f"{DATA_ROOT}/config.json"


def _load_config() -> Dict[str, Any]:
    """Load configuration from JSON file in data volume, provide defaults if missing."""
    defaults: Dict[str, Any] = {
        "model_name": "base",
        "language": "en",
        "temperature": 0.0,
        "word_timestamps": False,
        "device": "auto",
        "formats": ["txt", "json"],
        "batch_verbose": False,
    }
    try:
        config_file = Path(CONFIG_PATH)
        if config_file.exists():
            data = json.loads(config_file.read_text(encoding="utf-8"))
            defaults.update({k: v for k, v in data.items() if v is not None})
    except Exception as e:
        logger.warning(f"Failed to load config.json, using defaults: {e}")
    return defaults


def _ensure_dirs() -> None:
    for p in (DATA_ROOT, AUDIO_DIR, TRANSCRIPTS_DIR, MODEL_CACHE_DIR):
        Path(p).mkdir(parents=True, exist_ok=True)


@app.function(
    image=image,
    volumes={MODEL_CACHE_DIR: model_volume, DATA_ROOT: data_volume},
    gpu="T4",
    timeout=3600,
    memory=8192,
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
    # Lazy imports to avoid local ModuleNotFoundError during `modal serve`
    import whisper  # noqa: F401 (ensures model package is present in container)
    from vecina_transcriber.transcriber import EnglishTranscriber
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
    volumes={MODEL_CACHE_DIR: model_volume, DATA_ROOT: data_volume},
    gpu="T4",
    timeout=7200,
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
    import whisper  # noqa: F401
    from vecina_transcriber.transcriber import EnglishTranscriber
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
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=300,
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
        from vecina_transcriber.transcriber import EnglishTranscriber
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
    from vecina_transcriber.transcriber import EnglishTranscriber
    return EnglishTranscriber.get_available_models()


@app.function(
    image=image,
    volumes={MODEL_CACHE_DIR: model_volume, DATA_ROOT: data_volume},
    gpu="T4",
    timeout=3600,
    memory=8192,
)
def transcribe_single_file(audio_path_str: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe a single audio file (parallel worker function)."""
    from vecina_transcriber.transcriber import EnglishTranscriber
    import whisper  # noqa: F401

    audio_path = Path(audio_path_str)
    model_name = config["model_name"]

    try:
        logger.info(f"Transcribing {audio_path.name}")

        transcriber = EnglishTranscriber(
            model_name=model_name,
            download_root=MODEL_CACHE_DIR,
            device=None if config.get(
                "device") == "auto" else config.get("device")
        )

        result = transcriber.transcribe(
            audio=str(audio_path),
            temperature=config.get("temperature", 0.0),
            word_timestamps=config.get("word_timestamps", False),
            language=config.get("language", "en"),
            initial_prompt=config.get("initial_prompt"),
            verbose=config.get("batch_verbose", False)
        )

        result.update({
            "filename": audio_path.name,
            "model_used": model_name,
        })

        # Save outputs
        base_out = Path(TRANSCRIPTS_DIR) / audio_path.stem
        if "txt" in config.get("formats", []):
            (base_out.with_suffix(".txt")).write_text(
                result["text"], encoding="utf-8")
        if "json" in config.get("formats", []):
            (base_out.with_suffix(".json")).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        data_volume.commit()  # Commit after each file
        logger.info(f"Completed {audio_path.name}")
        return {"filename": audio_path.name, "status": "ok"}

    except Exception as e:
        logger.error(f"Failed {audio_path.name}: {e}")
        return {"filename": audio_path.name, "status": "error", "error": str(e)}


@app.function(
    image=image,
    volumes={MODEL_CACHE_DIR: model_volume, DATA_ROOT: data_volume},
    timeout=7200,
)
def transcribe_all_modal() -> Dict[str, Any]:
    """Transcribe all audio files in the data volume using parallel workers."""
    _ensure_dirs()
    cfg = _load_config()
    model_name = cfg["model_name"]
    max_workers = cfg.get("max_parallel_workers", 10)

    logger.info(
        f"Batch transcription start: model={model_name}, max_workers={max_workers}")

    audio_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    audio_files = [p for p in Path(AUDIO_DIR).rglob(
        "*") if p.is_file() and p.suffix.lower() in audio_exts]

    if not audio_files:
        logger.warning("No audio files found in volume.")
        return {"processed": 0, "results": []}

    logger.info(f"Found {len(audio_files)} audio file(s) to transcribe")

    # Prepare inputs for parallel map
    audio_paths_str = [str(p) for p in audio_files]
    configs = [cfg for _ in audio_files]

    # Run transcriptions in parallel using .map()
    results = list(transcribe_single_file.map(audio_paths_str, configs))

    summary = {"processed": len(results), "results": results}
    Path(TRANSCRIPTS_DIR, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    data_volume.commit()  # Final commit

    logger.info(
        f"Batch transcription complete: {len(results)} files processed")
    return summary


@app.function(image=image, volumes={DATA_ROOT: data_volume}, timeout=60)
def list_transcripts() -> Dict[str, Any]:
    """Web endpoint: list available transcript files."""
    _ensure_dirs()
    files = []
    for p in Path(TRANSCRIPTS_DIR).glob("*.*"):
        if p.is_file():
            files.append({"name": p.name, "size": p.stat().st_size})
    return {"count": len(files), "files": files}


@app.function(image=image, volumes={DATA_ROOT: data_volume}, timeout=60)
def get_transcript(filename: str) -> Dict[str, Any]:
    """Web endpoint: fetch a specific transcript (json or text)."""
    _ensure_dirs()
    target = Path(TRANSCRIPTS_DIR) / filename
    if not target.exists():
        return {"error": "not_found", "filename": filename}
    if target.suffix == ".json":
        return json.loads(target.read_text(encoding="utf-8"))
    # Return text wrapped in JSON
    return {"filename": filename, "text": target.read_text(encoding="utf-8")}


# FastAPI ASGI application providing richer endpoints.
@app.function(image=image, volumes={MODEL_CACHE_DIR: model_volume, DATA_ROOT: data_volume}, timeout=300)
@modal.asgi_app(requires_proxy_auth=True)
def api_app():
    """FastAPI application exposing transcript and batch operations."""
    from fastapi import FastAPI, HTTPException, Query

    _ensure_dirs()
    app_fast = FastAPI(title="VECINA Transcriber API", version="0.1.0")

    @app_fast.get("/health")
    async def health():
        return {"status": "ok"}

    @app_fast.get("/config")
    async def get_config():
        return _load_config()

    @app_fast.get("/transcripts")
    async def list_transcripts_api():
        files = []
        for p in Path(TRANSCRIPTS_DIR).glob("*.*"):
            if p.is_file():
                files.append({"name": p.name, "size": p.stat().st_size})
        return {"count": len(files), "files": files}

    @app_fast.get("/transcripts/{filename}")
    async def get_transcript_api(filename: str):
        target = Path(TRANSCRIPTS_DIR) / filename
        if not target.exists():
            raise HTTPException(status_code=404, detail="Transcript not found")
        if target.suffix == ".json":
            return json.loads(target.read_text(encoding="utf-8"))
        return {"filename": filename, "text": target.read_text(encoding="utf-8")}

    @app_fast.post("/batch")
    async def trigger_batch():
        # Fire synchronous batch transcription in current container
        summary = transcribe_all_modal.remote()
        return summary

    return app_fast


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
    # Delegate execution to the primary CLI defined in cli.py
    try:
        from .cli import main as cli_main
    except ImportError:  # Fallback if relative import fails
        from vecina_transcriber.cli import main as cli_main
    cli_main()
