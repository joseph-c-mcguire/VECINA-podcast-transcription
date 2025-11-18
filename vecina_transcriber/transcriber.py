"""
The purpose of this module is to transcribe English audio files using the Whisper model.

This module provides a high-level interface for transcribing English audio content
using OpenAI's Whisper model. It handles model loading, audio preprocessing,
and transcription with various quality control and customization options.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import whisper
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def _is_hallucination(segment: Dict[str, Any], compression_threshold: float = 2.4) -> bool:
    """
    Detect if a segment is likely a hallucination based on Whisper quality metrics.

    Args:
        segment: Segment dictionary from Whisper output
        compression_threshold: Maximum acceptable compression ratio

    Returns:
        True if segment appears to be a hallucination
    """
    # Check compression ratio (high ratio = repetitive text)
    if segment.get('compression_ratio', 0) > compression_threshold:
        return True

    # Check for very high no_speech probability (likely silence or noise)
    if segment.get('no_speech_prob', 0) > 0.8:
        return True

    # Check for extremely repetitive patterns in text
    text = segment.get('text', '').strip()
    if len(text) > 50:
        # Check if text has very repetitive 3-word phrases
        words = text.split()
        if len(words) > 10:
            # Count unique 3-word phrases
            phrases = set()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                phrases.add(phrase)
            # If less than 20% unique phrases, likely hallucination
            if len(phrases) / (len(words) - 2) < 0.2:
                return True

    return False


def _retry_segment_transcription(
    model: Any,
    audio: Union[str, Path],
    start_time: float,
    end_time: float,
    original_segment: Dict[str, Any],
    language: str = "en"
) -> Optional[Dict[str, Any]]:
    """
    Retry transcribing a segment with different parameters.

    Args:
        model: Whisper model instance
        audio: Path to audio file
        start_time: Segment start time in seconds
        end_time: Segment end time in seconds
        original_segment: Original segment dictionary
        language: Language code

    Returns:
        New segment if successful and better, None otherwise
    """
    try:
        from pydub import AudioSegment as AS

        # Extract the segment audio
        full_audio = AS.from_file(str(audio))
        segment_audio = full_audio[int(start_time * 1000):int(end_time * 1000)]

        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            segment_audio.export(tmp.name, format='wav')
            tmp_path = tmp.name

        try:
            # Retry with different parameters:
            # - Higher temperature for more variation
            # - No conditioning on previous text
            # - Lower compression ratio threshold
            retry_result = model.transcribe(
                audio=tmp_path,
                language=language,
                temperature=1.0,  # Higher temperature
                condition_on_previous_text=False,  # Don't use context
                compression_ratio_threshold=2.0,  # Stricter threshold
                no_speech_threshold=0.5,  # Lower silence threshold
                logprob_threshold=-0.5,  # Stricter confidence
                verbose=False
            )

            # Check if retry is better
            if retry_result and 'segments' in retry_result and retry_result['segments']:
                new_segment = retry_result['segments'][0]
                new_compression = new_segment.get(
                    'compression_ratio', float('inf'))
                old_compression = original_segment.get(
                    'compression_ratio', float('inf'))

                # Accept if compression ratio improved significantly
                if new_compression < old_compression * 0.7 and new_compression < 2.4:
                    logger.info(
                        f"Retry successful at {start_time:.2f}s: "
                        f"compression {old_compression:.2f} -> {new_compression:.2f}"
                    )
                    # Update timing to match original
                    new_segment['start'] = start_time
                    new_segment['end'] = end_time
                    new_segment['id'] = original_segment['id']
                    new_segment['retried'] = True
                    return new_segment
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.debug(f"Retry failed for segment at {start_time:.2f}s: {e}")

    return None


def _filter_hallucinations(
    result: Dict[str, Any],
    compression_threshold: float = 2.4,
    retry_hallucinations: bool = False,
    model: Any = None,
    audio: Union[str, Path, None] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Filter out hallucinated segments from transcription result.

    Args:
        result: Transcription result dictionary
        compression_threshold: Maximum acceptable compression ratio
        retry_hallucinations: Whether to retry hallucinated segments
        model: Whisper model instance (required if retry_hallucinations=True)
        audio: Audio file path (required if retry_hallucinations=True)
        language: Language code for retries

    Returns:
        Filtered result with hallucinations removed or retried
    """
    if 'segments' not in result:
        return result

    original_count = len(result['segments'])
    filtered_segments = []
    removed_segments = []
    retried_segments = []

    for segment in result['segments']:
        if _is_hallucination(segment, compression_threshold):
            # Try to retry if enabled and possible
            if retry_hallucinations and model is not None and audio is not None:
                retry_segment = _retry_segment_transcription(
                    model=model,
                    audio=audio,
                    start_time=segment.get('start', 0),
                    end_time=segment.get('end', 0),
                    original_segment=segment,
                    language=language
                )

                if retry_segment:
                    # Retry successful, use new segment
                    filtered_segments.append(retry_segment)
                    retried_segments.append(segment)
                    continue

            # No retry or retry failed - remove segment
            logger.warning(
                f"Removing hallucinated segment at {segment.get('start', 0):.2f}s: "
                f"compression_ratio={segment.get('compression_ratio', 0):.2f}, "
                f"no_speech_prob={segment.get('no_speech_prob', 0):.2f}"
            )
            removed_segments.append(segment)
        else:
            filtered_segments.append(segment)

    # Rebuild full text from filtered segments
    filtered_text = ' '.join(seg['text'].strip() for seg in filtered_segments)

    result['segments'] = filtered_segments
    result['text'] = filtered_text
    result['hallucination_filter'] = {
        'enabled': True,
        'retry_enabled': retry_hallucinations,
        'original_segment_count': original_count,
        'filtered_segment_count': len(filtered_segments),
        'removed_segment_count': len(removed_segments),
        'retried_segment_count': len(retried_segments),
        'compression_threshold': compression_threshold
    }

    if removed_segments or retried_segments:
        msg_parts = []
        if retried_segments:
            msg_parts.append(f"Retried {len(retried_segments)} segments")
        if removed_segments:
            msg_parts.append(f"Removed {len(removed_segments)} segments")
        logger.info(
            f"Hallucination filter: {', '.join(msg_parts)} "
            f"({(len(removed_segments) + len(retried_segments))/original_count*100:.1f}% of total)"
        )

    return result


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
        filter_hallucinations: bool = True,
        retry_hallucinations: bool = False,
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
            filter_hallucinations: Remove hallucinated segments (default: True)
            retry_hallucinations: Retry transcribing hallucinated segments with different parameters (default: False)
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

            # Filter hallucinations if enabled
            if filter_hallucinations:
                result = _filter_hallucinations(
                    result=result,
                    compression_threshold=compression_ratio_threshold or 2.4,
                    retry_hallucinations=retry_hallucinations,
                    model=self.model if retry_hallucinations else None,
                    audio=audio_input if retry_hallucinations and isinstance(
                        audio_input, (str, Path)) else None,
                    language=language
                )

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

    @staticmethod
    def split_audio_into_chunks(
        audio_file: Union[str, Path],
        chunk_duration_seconds: int = 600,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        Split an audio file into fixed-duration chunks.

        Args:
            audio_file: Path to the audio file to split
            chunk_duration_seconds: Duration of each chunk in seconds (default: 600 = 10 minutes)
            output_dir: Directory to save chunks (default: temp directory)

        Returns:
            List of paths to the chunk files
        """
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Loading audio file for chunking: {audio_path}")
        audio = AudioSegment.from_file(str(audio_path))

        total_duration_ms = len(audio)
        chunk_duration_ms = chunk_duration_seconds * 1000

        # Determine output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        chunk_files = []
        chunk_num = 0

        for start_ms in range(0, total_duration_ms, chunk_duration_ms):
            end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
            chunk = audio[start_ms:end_ms]

            chunk_filename = f"{audio_path.stem}_chunk_{chunk_num:04d}{audio_path.suffix}"
            chunk_path = output_dir / chunk_filename

            chunk.export(str(chunk_path), format=audio_path.suffix.lstrip('.'))
            chunk_files.append(chunk_path)

            logger.info(
                f"Created chunk {chunk_num}: {chunk_path} ({start_ms/1000:.1f}s - {end_ms/1000:.1f}s)")
            chunk_num += 1

        logger.info(f"Split audio into {len(chunk_files)} chunks")
        return chunk_files

    def transcribe_chunked(
        self,
        audio: Union[str, Path],
        chunk_duration_seconds: int = 600,
        cleanup_chunks: bool = True,
        filter_hallucinations: bool = True,
        retry_hallucinations: bool = False,
        **transcription_kwargs: Any
    ) -> Dict:
        """
        Transcribe a long audio file by splitting it into chunks first.
        This improves accuracy for long recordings.

        Args:
            audio: Path to the audio file
            chunk_duration_seconds: Duration of each chunk in seconds (default: 600 = 10 minutes)
            cleanup_chunks: Whether to delete chunk files after transcription (default: True)
            filter_hallucinations: Remove hallucinated segments (default: True)
            retry_hallucinations: Retry transcribing hallucinated segments (default: False)
            **transcription_kwargs: Additional arguments passed to transcribe()

        Returns:
            Dictionary containing:
            - "text": Full merged transcription
            - "segments": List of all segments with adjusted timestamps
            - "chunks": List of chunk-level metadata
            - "language": Language code
        """
        audio_path = Path(audio)
        logger.info(f"Starting chunked transcription of {audio_path}")
        logger.info(f"Chunk duration: {chunk_duration_seconds} seconds")

        # Split audio into chunks
        chunk_files = self.split_audio_into_chunks(
            audio_file=audio_path,
            chunk_duration_seconds=chunk_duration_seconds
        )

        try:
            all_text = []
            all_segments = []
            chunk_metadata = []
            time_offset = 0.0
            detected_language = None

            # Transcribe each chunk
            for idx, chunk_file in enumerate(chunk_files):
                logger.info(
                    f"Transcribing chunk {idx + 1}/{len(chunk_files)}: {chunk_file.name}")

                result = self.transcribe(
                    audio=chunk_file,
                    filter_hallucinations=filter_hallucinations,
                    retry_hallucinations=retry_hallucinations,
                    **transcription_kwargs
                )

                # Store detected language from first chunk
                if detected_language is None:
                    detected_language = result.get("language", "en")

                # Collect text
                all_text.append(result["text"])

                # Adjust segment timestamps and collect
                for segment in result.get("segments", []):
                    adjusted_segment = segment.copy()
                    adjusted_segment["start"] += time_offset
                    adjusted_segment["end"] += time_offset
                    adjusted_segment["chunk_index"] = idx
                    all_segments.append(adjusted_segment)

                # Store chunk metadata
                chunk_metadata.append({
                    "chunk_index": idx,
                    "chunk_file": chunk_file.name,
                    "time_offset": time_offset,
                    "duration": result.get("segments", [])[-1]["end"] if result.get("segments") else 0.0,
                    "text_length": len(result["text"])
                })

                # Update time offset for next chunk
                if result.get("segments"):
                    time_offset += result["segments"][-1]["end"]

            # Merge results
            merged_result = {
                "text": " ".join(all_text),
                "segments": all_segments,
                "chunks": chunk_metadata,
                "language": detected_language,
                "num_chunks": len(chunk_files),
                "chunk_duration_seconds": chunk_duration_seconds
            }

            logger.info(
                f"Chunked transcription complete. Total text length: {len(merged_result['text'])}")
            return merged_result

        finally:
            # Cleanup chunk files if requested
            if cleanup_chunks:
                for chunk_file in chunk_files:
                    try:
                        chunk_file.unlink()
                        logger.debug(f"Deleted chunk file: {chunk_file}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete chunk file {chunk_file}: {e}")

                # Try to remove the temp directory
                try:
                    chunk_dir = chunk_files[0].parent
                    if chunk_dir.exists() and not list(chunk_dir.iterdir()):
                        chunk_dir.rmdir()
                        logger.debug(f"Deleted temp directory: {chunk_dir}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp directory: {e}")


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
