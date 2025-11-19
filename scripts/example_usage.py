#!/usr/bin/env python3
"""
Example usage script for the VECINA podcast transcription tool.

This script demonstrates various ways to use the transcription functionality,
including local processing and Modal deployment.
"""

from vecina_transcriber import (
    EnglishTranscriber,
    create_transcriber,
    transcribe_audio,
    is_modal_available,
    get_package_info,
)
import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import vecina_transcriber
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Demonstrate basic transcription usage."""
    print("=" * 50)
    print("Basic Transcription Example")
    print("=" * 50)

    # This would work with an actual audio file
    audio_file = "sample_audio.wav"  # Replace with actual file path

    if not Path(audio_file).exists():
        print(f"⚠️  Audio file {audio_file} not found. Skipping this example.")
        return

    try:
        # Method 1: Using the convenience function
        print("Method 1: Using convenience function")
        result = transcribe_audio(audio_file, model_name="tiny")
        print(f"Transcription (first 100 chars): {result['text'][:100]}...")

        # Method 2: Using the transcriber clas
        print("\nMethod 2: Using transcriber class")
        transcriber = create_transcriber(model_name="tiny")
        result = transcriber.transcribe_file(
            audio_file, output_file="output.txt")
        print(f"Text saved to output.txt")

        # Method 3: Direct class instantiation with custom settings
        print("\nMethod 3: Custom transcriber settings")
        transcriber = EnglishTranscriber(
            model_name="base",
            device="cpu"  # Force CPU usage
        )

        result = transcriber.transcribe(
            audio_file,
            temperature=0.2,
            initial_prompt="This is a podcast episode discussing technology.",
            word_timestamps=True
        )

        print(f"Segments found: {len(result['segments'])}")
        if result['segments']:
            first_segment = result['segments'][0]
            print(f"First segment: {first_segment['text'][:50]}...")

        # Method 4: Chunked transcription for long files
        print("\nMethod 4: Chunked transcription (better accuracy for long files)")
        transcriber = create_transcriber(model_name="base")
        result = transcriber.transcribe_chunked(
            audio_file,
            chunk_duration_seconds=300,  # 5-minute chunks
            temperature=0.0,
            initial_prompt="This is a podcast episode."
        )

        print(f"Chunked transcription complete!")
        print(f"Total chunks: {result.get('num_chunks', 0)}")
        print(f"Total text length: {len(result['text'])} characters")

    except Exception as e:
        print(f"❌ Error in basic usage: {e}")


def example_model_info():
    """Demonstrate model information retrieval."""
    print("\n" + "=" * 50)
    print("Model Information Example")
    print("=" * 50)

    try:
        # List available models
        print("Available Whisper models:")
        models = EnglishTranscriber.get_available_models()
        for model in models:
            print(f"  - {model}")

        # Get detailed model info
        print(f"\nDetailed info for 'tiny' model:")
        transcriber = EnglishTranscriber(model_name="tiny")
        model_info = transcriber.get_model_info()

        for key, value in model_info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")

    except Exception as e:
        print(f"❌ Error getting model info: {e}")


def example_batch_processing():
    """Demonstrate batch processing of multiple audio files."""
    print("\n" + "=" * 50)
    print("Batch Processing Example")
    print("=" * 50)

    # Sample audio files (replace with actual files)
    audio_files = [
        "episode1.wav",
        "episode2.wav",
        "episode3.wav"
    ]

    # Filter to only existing files
    existing_files = [f for f in audio_files if Path(f).exists()]

    if not existing_files:
        print("⚠️  No audio files found for batch processing. Skipping this example.")
        return

    try:
        # Create a single transcriber instance for efficiency
        transcriber = create_transcriber(model_name="tiny")

        results = []
        for audio_file in existing_files:
            print(f"Processing {audio_file}...")
            result = transcriber.transcribe_file(
                audio_file,
                output_file=f"transcription_{Path(audio_file).stem}.txt"
            )
            results.append({
                "file": audio_file,
                "text_length": len(result['text']),
                "segments": len(result['segments'])
            })

        # Summary
        print("\nBatch Processing Summary:")
        for result in results:
            print(
                f"  {result['file']}: {result['text_length']} chars, {result['segments']} segments")

    except Exception as e:
        print(f"❌ Error in batch processing: {e}")


def example_modal_usage():
    """Demonstrate Modal deployment usage (if available)."""
    print("\n" + "=" * 50)
    print("Modal Deployment Example")
    print("=" * 50)

    if not is_modal_available():
        print(
            "⚠️  Modal is not available. Install with: pip install 'vecina-transcriber[modal]'")
        return

    try:
        from vecina_transcriber import transcribe_with_modal, batch_transcribe_with_modal

        # This would work with actual audio files and Modal setup
        audio_file = "sample_audio.wav"

        if not Path(audio_file).exists():
            print(
                f"⚠️  Audio file {audio_file} not found. Skipping Modal example.")
            return

        print("Transcribing using Modal (cloud processing)...")
        result = transcribe_with_modal(
            audio_file,
            model_name="base",
            temperature=0.1,
            chunk_duration_seconds=600  # Use 10-minute chunks for long files
        )

        print(f"Modal transcription completed!")
        print(f"Text (first 100 chars): {result['text'][:100]}...")
        print(f"Processed in Modal: {result.get('processed_in_modal', False)}")
        if result.get('num_chunks'):
            print(f"Chunks processed: {result.get('num_chunks')}")

    except Exception as e:
        print(f"❌ Error with Modal usage: {e}")


def example_output_formats():
    """Demonstrate different output formats."""
    print("\n" + "=" * 50)
    print("Output Formats Example")
    print("=" * 50)

    audio_file = "sample_audio.wav"

    if not Path(audio_file).exists():
        print(
            f"⚠️  Audio file {audio_file} not found. Skipping format examples.")
        return

    try:
        transcriber = create_transcriber(model_name="tiny")

        # Get base transcription with word timestamps
        result = transcriber.transcribe(
            audio_file,
            word_timestamps=True
        )

        # Simulate saving different formats (normally done by CLI)
        formats = {
            "text": result['text'],
            "segments_count": len(result['segments']),
            "has_word_timestamps": any('words' in seg for seg in result['segments'])
        }

        print("Output format capabilities:")
        for format_name, info in formats.items():
            print(f"  {format_name}: {info}")

        print("\nSupported output formats:")
        print("  - text: Plain text transcription")
        print("  - json: Full result with metadata")
        print("  - srt: Subtitle format")
        print("  - vtt: WebVTT format")

    except Exception as e:
        print(f"❌ Error demonstrating formats: {e}")


def main():
    """Run all examples."""
    print("VECINA Podcast Transcription Tool - Example Usage")
    print("=" * 60)

    # Package info
    package_info = get_package_info()
    print(f"Package: {package_info['name']} v{package_info['version']}")
    print(f"Modal available: {package_info['modal_available']}")
    print(f"Description: {package_info['description']}")

    # Run examples
    example_model_info()
    example_basic_usage()
    example_batch_processing()
    example_output_formats()
    example_modal_usage()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo use the CLI tool:")
    print("  vecina-transcriber transcribe your_audio_file.wav")
    print("  vecina-transcriber models")
    print("  vecina-transcriber info your_audio_file.wav")


if __name__ == "__main__":
    main()
