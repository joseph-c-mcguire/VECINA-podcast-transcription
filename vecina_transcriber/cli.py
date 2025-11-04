"""
This is the command line interface for the VECINA podcast transcription tool.

The main command is for transcribing audio files using various models and settings.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .transcriber import EnglishTranscriber, transcribe_audio


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool) -> None:
    """VECINA Podcast Transcription Tool - Transcribe audio using Whisper models."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--model', '-m',
    default='base',
    type=click.Choice(['tiny', 'base', 'small', 'medium',
                      'large', 'large-v2', 'large-v3']),
    help='Whisper model to use for transcription'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output file path for transcription text'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['text', 'json', 'srt', 'vtt']),
    default='text',
    help='Output format'
)
@click.option(
    '--device',
    type=click.Choice(['cpu', 'cuda', 'auto']),
    default='auto',
    help='Device to run the model on'
)
@click.option(
    '--temperature',
    type=float,
    default=0.0,
    help='Sampling temperature (0.0 = deterministic)'
)
@click.option(
    '--initial-prompt',
    help='Initial prompt to guide the transcription'
)
@click.option(
    '--word-timestamps',
    is_flag=True,
    help='Extract word-level timestamps'
)
@click.option(
    '--language',
    default='en',
    help='Language code (default: en for English)'
)
@click.option(
    '--no-speech-threshold',
    type=float,
    default=0.6,
    help='Threshold for detecting silence'
)
@click.option(
    '--compression-ratio-threshold',
    type=float,
    default=2.4,
    help='Threshold for detecting repetitive text'
)
@click.option(
    '--logprob-threshold',
    type=float,
    default=-1.0,
    help='Threshold for detecting low-confidence segments'
)
def transcribe(
    audio_file: Path,
    model: str,
    output: Optional[Path],
    format: str,
    device: str,
    temperature: float,
    initial_prompt: Optional[str],
    word_timestamps: bool,
    language: str,
    no_speech_threshold: float,
    compression_ratio_threshold: float,
    logprob_threshold: float
) -> None:
    """Transcribe an audio file using Whisper."""
    try:
        # Auto-detect device if specified
        device_name = None if device == 'auto' else device

        logger.info(f"Starting transcription of {audio_file}")
        logger.info(f"Using model: {model}, device: {device}")

        # Create transcriber instance
        transcriber = EnglishTranscriber(
            model_name=model,
            device=device_name
        )

        # Transcribe the audio
        result = transcriber.transcribe(
            audio=audio_file,
            temperature=temperature,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            language=language,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            verbose=True
        )

        # Generate output filename if not provided
        if output is None:
            output = audio_file.with_suffix(f'.{format}')

        # Save result based on format
        _save_transcription(result, output, format)

        logger.info(f"Transcription completed successfully")
        logger.info(f"Output saved to: {output}")

        # Print summary
        click.echo(f"\n✅ Transcription completed!")
        click.echo(f"📁 Input: {audio_file}")
        click.echo(f"💾 Output: {output}")
        click.echo(f"🤖 Model: {model}")
        click.echo(f"📝 Text length: {len(result['text'])} characters")
        click.echo(f"⏱️  Segments: {len(result['segments'])}")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def models() -> None:
    """List available Whisper models."""
    available_models = EnglishTranscriber.get_available_models()

    click.echo("Available Whisper models:")
    click.echo("=" * 40)

    model_info = {
        'tiny': 'Fastest, least accurate (~32MB)',
        'base': 'Good balance of speed/accuracy (~74MB)',
        'small': 'Better accuracy, slower (~244MB)',
        'medium': 'Good accuracy, moderate speed (~769MB)',
        'large': 'Best accuracy, slowest (~1550MB)',
        'large-v2': 'Improved large model (~1550MB)',
        'large-v3': 'Latest large model (~1550MB)'
    }

    for model in available_models:
        description = model_info.get(model, 'Model description not available')
        click.echo(f"  {model:<12} - {description}")


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True, path_type=Path))
@click.option('--model', '-m', default='base', help='Model to use for info extraction')
def info(audio_file: Path, model: str) -> None:
    """Get information about an audio file and test model loading."""
    try:
        # Create transcriber to test model loading
        transcriber = EnglishTranscriber(model_name=model)
        model_info = transcriber.get_model_info()

        click.echo(f"Audio file: {audio_file}")
        click.echo(f"File exists: {audio_file.exists()}")
        click.echo(
            f"File size: {audio_file.stat().st_size / (1024*1024):.2f} MB")
        click.echo()
        click.echo("Model Information:")
        click.echo("=" * 30)
        for key, value in model_info.items():
            if isinstance(value, dict):
                click.echo(f"{key}:")
                for sub_key, sub_value in value.items():
                    click.echo(f"  {sub_key}: {sub_value}")
            else:
                click.echo(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Info command failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def _save_transcription(result: dict, output_path: Path, format: str) -> None:
    """Save transcription result in the specified format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'text':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])

    elif format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    elif format == 'srt':
        _save_srt(result, output_path)

    elif format == 'vtt':
        _save_vtt(result, output_path)


def _save_srt(result: dict, output_path: Path) -> None:
    """Save transcription as SRT subtitle format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], 1):
            start_time = _format_timestamp_srt(segment['start'])
            end_time = _format_timestamp_srt(segment['end'])
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def _save_vtt(result: dict, output_path: Path) -> None:
    """Save transcription as WebVTT format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for segment in result['segments']:
            start_time = _format_timestamp_vtt(segment['start'])
            end_time = _format_timestamp_vtt(segment['end'])
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def _format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for WebVTT format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"


def main() -> None:
    """Main entry point for the CLI."""
    cli()
