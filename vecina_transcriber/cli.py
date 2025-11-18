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
@click.option(
    '--chunk-duration',
    type=int,
    help='Split audio into chunks of this duration (seconds) for better accuracy on long files'
)
@click.option(
    '--filter-hallucinations/--no-filter-hallucinations',
    default=True,
    help='Filter out hallucinated/repetitive segments (enabled by default)'
)
@click.option(
    '--retry-hallucinations',
    is_flag=True,
    help='Retry transcribing hallucinated segments with different parameters before removing them'
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
    logprob_threshold: float,
    chunk_duration: Optional[int],
    filter_hallucinations: bool,
    retry_hallucinations: bool
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

        # Transcribe the audio (chunked or standard)
        if chunk_duration:
            logger.info(
                f"Using chunked transcription with {chunk_duration}s chunks")
            result = transcriber.transcribe_chunked(
                audio=audio_file,
                chunk_duration_seconds=chunk_duration,
                filter_hallucinations=filter_hallucinations,
                retry_hallucinations=retry_hallucinations,
                temperature=temperature,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                language=language,
                no_speech_threshold=no_speech_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                verbose=True
            )
        else:
            result = transcriber.transcribe(
                audio=audio_file,
                filter_hallucinations=filter_hallucinations,
                retry_hallucinations=retry_hallucinations,
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
        if chunk_duration:
            click.echo(
                f"🧩 Chunks: {result.get('num_chunks', 0)} ({chunk_duration}s each)")

        # Show hallucination filter stats if available
        if 'hallucination_filter' in result and filter_hallucinations:
            filter_info = result['hallucination_filter']
            removed = filter_info.get('removed_segment_count', 0)
            retried = filter_info.get('retried_segment_count', 0)
            if retried > 0:
                click.echo(f"🔄 Retried {retried} hallucinated segments")
            if removed > 0:
                click.echo(f"🗑️  Removed {removed} hallucinated segments")
            if removed == 0 and retried == 0:
                click.echo(f"🔍 No hallucinations detected")

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    help='Output directory for transcriptions'
)
@click.option(
    '--model', '-m',
    default='base',
    type=click.Choice(['tiny', 'base', 'small', 'medium',
                      'large', 'large-v2', 'large-v3']),
    help='Whisper model to use for transcription'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['text', 'json', 'both']),
    default='both',
    help='Output format(s)'
)
@click.option(
    '--chunk-duration',
    type=int,
    help='Split audio into chunks of this duration (seconds) for better accuracy'
)
@click.option(
    '--filter-hallucinations/--no-filter-hallucinations',
    default=True,
    help='Filter out hallucinated/repetitive segments'
)
@click.option(
    '--retry-hallucinations',
    is_flag=True,
    help='Retry transcribing hallucinated segments before removing them'
)
@click.option(
    '--pattern',
    default='*.mp3',
    help='File pattern to match (e.g., "*.mp3", "*.wav")'
)
def batch(
    input_dir: Path,
    output_dir: Optional[Path],
    model: str,
    format: str,
    chunk_duration: Optional[int],
    filter_hallucinations: bool,
    retry_hallucinations: bool,
    pattern: str
) -> None:
    """Batch transcribe all audio files in a directory."""
    try:
        import glob

        # Find all matching audio files
        audio_files = list(input_dir.glob(pattern))

        if not audio_files:
            click.echo(f"❌ No audio files found matching pattern: {pattern}")
            sys.exit(1)

        # Set output directory
        if output_dir is None:
            output_dir = input_dir / '_output'
        output_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"🎙️  Found {len(audio_files)} audio files to transcribe")
        click.echo(f"📁 Output directory: {output_dir}")
        click.echo(f"🤖 Model: {model}")
        if chunk_duration:
            click.echo(f"🧩 Chunk duration: {chunk_duration}s")
        click.echo(
            f"🔍 Hallucination filtering: {'enabled' if filter_hallucinations else 'disabled'}")
        if retry_hallucinations:
            click.echo(f"🔄 Retry hallucinations: enabled")
        click.echo()

        # Create transcriber once for all files
        transcriber = EnglishTranscriber(model_name=model)

        success_count = 0
        failed_files = []

        for idx, audio_file in enumerate(audio_files, 1):
            try:
                click.echo(
                    f"[{idx}/{len(audio_files)}] Transcribing: {audio_file.name}")

                # Transcribe (chunked or standard)
                if chunk_duration:
                    result = transcriber.transcribe_chunked(
                        audio=audio_file,
                        chunk_duration_seconds=chunk_duration,
                        filter_hallucinations=filter_hallucinations,
                        retry_hallucinations=retry_hallucinations,
                        verbose=False
                    )
                else:
                    result = transcriber.transcribe(
                        audio=audio_file,
                        filter_hallucinations=filter_hallucinations,
                        retry_hallucinations=retry_hallucinations,
                        verbose=False
                    )

                # Save outputs
                base_name = audio_file.stem

                if format in ['text', 'both']:
                    text_output = output_dir / f"{base_name}.txt"
                    with open(text_output, 'w', encoding='utf-8') as f:
                        f.write(result['text'])
                    click.echo(f"  💾 Saved: {text_output.name}")

                if format in ['json', 'both']:
                    json_output = output_dir / f"{base_name}.json"
                    with open(json_output, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    click.echo(f"  💾 Saved: {json_output.name}")

                # Show hallucination filter stats if available
                if 'hallucination_filter' in result:
                    filter_info = result['hallucination_filter']
                    removed = filter_info.get('removed_segment_count', 0)
                    retried = filter_info.get('retried_segment_count', 0)
                    if retried > 0:
                        click.echo(f"  🔄 Retried {retried} segments")
                    if removed > 0:
                        click.echo(f"  🗑️  Removed {removed} segments")

                success_count += 1
                click.echo(
                    f"  ✅ Complete ({len(result['text'])} chars, {len(result['segments'])} segments)")
                click.echo()

            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file.name}: {e}")
                click.echo(f"  ❌ Error: {e}", err=True)
                failed_files.append((audio_file.name, str(e)))
                click.echo()

        # Summary
        click.echo("=" * 60)
        click.echo(f"✅ Batch transcription complete!")
        click.echo(
            f"📊 Successfully transcribed: {success_count}/{len(audio_files)}")

        if failed_files:
            click.echo(f"❌ Failed: {len(failed_files)}")
            click.echo("\nFailed files:")
            for filename, error in failed_files:
                click.echo(f"  - {filename}: {error}")

        click.echo(f"\n💾 Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"Batch transcription failed: {e}")
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
