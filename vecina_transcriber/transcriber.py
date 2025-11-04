"""
The purpose of this module is to transcribe English audio files using the Whisper model.

This module provides a high-level interface for transcribing English audio content
using OpenAI's Whisper model. It handles model loading, audio preprocessing,
and transcription with various quality control and customization options.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import whisper

logger = logging.getLogger(__name__)


class EnglishTranscriber:
    """
    A transcriber class for English audio using OpenAI's Whisper model.

    This class provides methods to load Whisper models and transcribe English
    audio files with various configuration options for quality control and
    output formatting.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        download_root: Optional[str] = None
    ):
        """
        Initialize the English transcriber.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            download_root: Path to download models to (None for default)
        """
        self.model_name = model_name
        self.device = device
        self.download_root = download_root
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            logger.info("Loading Whisper model: %s", self.model_name)
            self.model = whisper.load_model(
                name=self.model_name,
                device=self.device,
                download_root=self.download_root
            )
            logger.info("Model %s loaded successfully", self.model_name)
        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, e)
            raise

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        *,
        verbose: Optional[bool] = None,
        temperature: Union[float, Tuple[float, ...]] = (
            0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        carry_initial_prompt: bool = False,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：)]}、",
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        language: str = "en",
        **decode_options,
    ) -> Dict:
        """
        Transcribe audio using the Whisper model.

        Args:
            audio: Audio file path, numpy array, or torch tensor
            verbose: Controls progress display (True, False, or None)
            temperature: Sampling temperature(s) for fallback strategy
            compression_ratio_threshold: Detect repetitive output (default: 2.4)
            logprob_threshold: Detect low-confidence output (default: -1.0)
            no_speech_threshold: Detect silent segments (default: 0.6)
            condition_on_previous_text: Use previous output as context
            initial_prompt: Custom vocabulary or context prompt
            carry_initial_prompt: Prepend initial prompt to each segment
            word_timestamps: Extract word-level timing information
            prepend_punctuations: Punctuation handling for word timestamps
            append_punctuations: Punctuation handling for word timestamps
            clip_timestamps: Process specific time ranges ("start,end,start,end,...")
            hallucination_silence_threshold: Skip silent periods in word timestamps
            language: Language code (default: "en" for English)
            **decode_options: Additional parameters passed to DecodingOptions

        Returns:
            Dictionary containing:
            - "text": Full transcribed text
            - "segments": List of segment dictionaries with timing and metadata
            - "language": Detected or specified language code

        Raises:
            FileNotFoundError: If audio file path doesn't exist
            ValueError: If audio format is not supported
            RuntimeError: If transcription fails
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Please initialize the transcriber first.")

        # Validate audio input
        if isinstance(audio, (str, Path)):
            audio_path = Path(audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            logger.info("Transcribing audio file: %s", audio_path)
            # Convert Path to string for whisper compatibility
            audio_input = str(audio_path)
        else:
            logger.info("Transcribing audio from array/tensor")
            audio_input = audio

        try:
            # Set default initial prompt for English if not provided
            if initial_prompt is None:
                initial_prompt = "The following is a podcast episode in English."

            result = whisper.transcribe(
                model=self.model,
                audio=audio_input,
                verbose=verbose,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                carry_initial_prompt=carry_initial_prompt,
                word_timestamps=word_timestamps,
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations,
                clip_timestamps=clip_timestamps,
                hallucination_silence_threshold=hallucination_silence_threshold,
                language=language,
                **decode_options,
            )

            logger.info("Transcription completed. Text length: %d",
                        len(result['text']))
            return result

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            raise RuntimeError(f"Failed to transcribe audio: {e}") from e

    def transcribe_file(
        self,
        audio_file: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Dict:
        """
        Transcribe an audio file and optionally save the result.

        Args:
            audio_file: Path to the audio file
            output_file: Optional path to save the transcription text
            **kwargs: Additional arguments passed to transcribe()

        Returns:
            Transcription result dictionary
        """
        result = self.transcribe(audio_file, **kwargs)

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])

            logger.info("Transcription saved to: %s", str(output_path))

        return result

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"model_name": self.model_name, "loaded": False}

        return {
            "model_name": self.model_name,
            "loaded": True,
            "device": str(self.model.device),
            "is_multilingual": self.model.is_multilingual,
            "dims": {
                "n_mels": self.model.dims.n_mels,
                "n_audio_ctx": self.model.dims.n_audio_ctx,
                "n_audio_state": self.model.dims.n_audio_state,
                "n_audio_head": self.model.dims.n_audio_head,
                "n_audio_layer": self.model.dims.n_audio_layer,
                "n_vocab": self.model.dims.n_vocab,
                "n_text_ctx": self.model.dims.n_text_ctx,
                "n_text_state": self.model.dims.n_text_state,
                "n_text_head": self.model.dims.n_text_head,
                "n_text_layer": self.model.dims.n_text_layer,
            }
        }

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available Whisper model names.

        Returns:
            List of model names
        """
        return list(whisper.available_models())


def create_transcriber(
    model_name: str = "base",
    device: Optional[str] = None,
    download_root: Optional[str] = None
) -> EnglishTranscriber:
    """
    Factory function to create an English transcriber instance.

    Args:
        model_name: Whisper model name
        device: Device to run the model on
        download_root: Path to download models to

    Returns:
        EnglishTranscriber instance
    """
    return EnglishTranscriber(
        model_name=model_name,
        device=device,
        download_root=download_root
    )


def transcribe_audio(
    audio_file: Union[str, Path],
    model_name: str = "base",
    output_file: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> Dict:
    """
    Convenience function to transcribe an audio file with minimal setup.

    Args:
        audio_file: Path to the audio file
        model_name: Whisper model name to use
        output_file: Optional path to save the transcription text
        **kwargs: Additional arguments passed to transcribe()

    Returns:
        Transcription result dictionary
    """
    transcriber = create_transcriber(model_name=model_name)
    return transcriber.transcribe_file(audio_file, output_file, **kwargs)
