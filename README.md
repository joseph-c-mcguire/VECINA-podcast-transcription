# VECINA Podcast Transcription Tool

A comprehensive toolkit for transcribing podcast audio using OpenAI's Whisper model. Features both local processing and scalable cloud deployment via Modal.

## Features

### Core Transcription

- **Easy-to-use interface**: Simple class-based API for transcribing audio files
- **Multiple model sizes**: Support for all Whisper model sizes (tiny, base, small, medium, large, large-v2, large-v3)
- **Quality control**: Built-in parameters for controlling transcription quality and handling edge cases
- **Word-level timestamps**: Optional word-level timing information
- **Flexible input**: Accepts file paths, numpy arrays, or torch tensors
- **Multiple output formats**: Text, JSON, SRT, WebVTT

### Command Line Interface

- **Full CLI tool**: Complete command-line interface with `vecina-transcriber` command
- **Batch processing**: Process multiple files efficiently
- **Model information**: Get details about available models
- **Flexible configuration**: Extensive options for fine-tuning transcription

### Cloud Deployment

- **Modal integration**: Scale to cloud processing with Modal deployment
- **GPU acceleration**: Automatic GPU usage for faster processing
- **Batch cloud processing**: Process multiple files in parallel
- **Model caching**: Persistent model storage for faster subsequent runs

## Installation

### Prerequisites

**Important**: You need to install `ffmpeg` for audio processing:

#### Windows:
```powershell
# Using Chocolatey (recommended)
choco install ffmpeg

# Or using Winget
winget install ffmpeg
```

#### macOS:
```bash
brew install ffmpeg
```

#### Linux:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

### Basic Installation

```bash
pip install -e .
```

This installs the core transcription functionality with local processing capabilities.

### With Modal Support (Cloud Processing)

```bash
pip install -e ".[modal]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

The easiest way to get started is with the command-line tool:

```bash
# Transcribe a single file
vecina-transcriber transcribe audio_file.wav

# Use a specific model
vecina-transcriber transcribe audio_file.wav --model base

# Save to specific output file
vecina-transcriber transcribe audio_file.wav --output transcription.txt

# Get model information
vecina-transcriber models

# Get audio file info
vecina-transcriber info audio_file.wav
```

### Python API

## Quick Start

### Basic Usage

```python
from vecina_transcriber.english_transcriber import create_transcriber

# Create a transcriber with the base model
transcriber = create_transcriber(model_name="base")

# Transcribe an audio file
result = transcriber.transcribe("path/to/audio/file.mp3")
print(result['text'])
```

### Convenience Function

```python
from vecina_transcriber.english_transcriber import transcribe_audio

# Quick transcription with automatic output saving
result = transcribe_audio(
    audio_file="podcast_episode.mp3",
    model_name="base",
    output_file="transcription.txt"
)
```

## Advanced Usage

### Custom Configuration

```python
from vecina_transcriber.english_transcriber import EnglishTranscriber

# Create transcriber with custom settings
transcriber = EnglishTranscriber(
    model_name="small",  # Better quality than base
    device="cuda",       # Use GPU if available
    download_root="/path/to/models"  # Custom model storage location
)

# Transcribe with advanced parameters
result = transcriber.transcribe(
    audio="podcast.wav",
    word_timestamps=True,  # Include word-level timing
    initial_prompt="This is a technology podcast episode.",
    temperature=0.0,       # More deterministic output
    compression_ratio_threshold=2.4,  # Detect repetitive content
    logprob_threshold=-1.0,            # Filter low-confidence segments
    no_speech_threshold=0.6            # Handle silent segments
)
```

### File Operations

```python
# Transcribe and save to file
result = transcriber.transcribe_file(
    audio_file="input.mp3",
    output_file="output.txt",
    word_timestamps=True
)

# Get model information
info = transcriber.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Device: {info['device']}")
print(f"Multilingual: {info['is_multilingual']}")
```

## Model Selection

Choose the appropriate model size based on your needs:

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `tiny` | 39 MB | Very Fast | Basic | Quick testing, low-resource environments |
| `base` | 74 MB | Fast | Good | General use, balanced speed/quality |
| `small` | 244 MB | Medium | Better | High-quality transcription |
| `medium` | 769 MB | Slow | High | Professional transcription |
| `large` | 1550 MB | Very Slow | Highest | Maximum accuracy required |

```python
# Get list of available models
from vecina_transcriber.english_transcriber import EnglishTranscriber
models = EnglishTranscriber.get_available_models()
print(models)
```

## Quality Control Parameters

The transcriber provides several parameters to control output quality:

### Detection Thresholds

- `compression_ratio_threshold` (default: 2.4): Detects repetitive or looped content
- `logprob_threshold` (default: -1.0): Filters segments with low confidence scores
- `no_speech_threshold` (default: 0.6): Handles silent or non-speech segments
- `hallucination_silence_threshold`: Skips silent periods in word timestamps

### Text Processing

- `condition_on_previous_text` (default: True): Uses previous output as context
- `initial_prompt`: Provides vocabulary hints or context (e.g., "This is a technology podcast")
- `carry_initial_prompt`: Prepends initial prompt to each segment

### Timestamps and Segmentation

- `word_timestamps` (default: False): Extracts word-level timing information
- `clip_timestamps`: Process specific time ranges ("start,end,start,end,...")
- `prepend_punctuations`/`append_punctuations`: Controls punctuation handling

## Output Format

The transcribe method returns a dictionary with:

```python
{
    "text": "Full transcribed text as a single string",
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 5.5,
            "text": " This is the first segment.",
            "tokens": [50364, 50365, ...],
            "temperature": 0.0,
            "avg_logprob": -0.45,
            "compression_ratio": 1.2,
            "no_speech_prob": 0.01,
            "words": [  # Only if word_timestamps=True
                {
                    "word": "This",
                    "start": 0.0,
                    "end": 0.3,
                    "probability": 0.99
                },
                ...
            ]
        },
        ...
    ],
    "language": "en"
}
```

## Error Handling

The transcriber includes comprehensive error handling:

```python
try:
    result = transcriber.transcribe("nonexistent_file.mp3")
except FileNotFoundError as e:
    print(f"Audio file not found: {e}")
except RuntimeError as e:
    print(f"Transcription failed: {e}")
```

## Logging

The module uses Python's standard logging. Configure logging to see transcription progress:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now you'll see progress messages like:
# INFO:vecina_transcriber.english_transcriber:Loading Whisper model: base
# INFO:vecina_transcriber.english_transcriber:Model base loaded successfully
# INFO:vecina_transcriber.english_transcriber:Transcribing audio file: podcast.mp3
# INFO:vecina_transcriber.english_transcriber:Transcription completed. Text length: 1500
```

## Performance Tips

1. **Model Selection**: Use the smallest model that meets your quality needs
2. **GPU Acceleration**: Set `device="cuda"` if you have a compatible GPU
3. **Batch Processing**: Reuse the same transcriber instance for multiple files
4. **Memory Management**: For very long audio files, consider splitting them first
5. **Quality vs Speed**: Adjust temperature and threshold parameters for your use case

## Integration with VECINA Pipeline

This module is designed to integrate seamlessly with the VECINA podcast transcription pipeline:

```python
# Example integration
from vecina_transcriber.english_transcriber import create_transcriber
from vecina_transcriber.audio_getters.mp3_getter import get_mp3_from_url

# Download audio
audio_file = get_mp3_from_url("https://example.com/podcast.mp3")

# Transcribe
transcriber = create_transcriber(model_name="base")
result = transcriber.transcribe_file(
    audio_file=audio_file,
    output_file="transcription.txt",
    initial_prompt="This is a podcast episode."
)

print(f"Transcribed {len(result['text'])} characters")
```

## API Reference

### EnglishTranscriber Class

#### Constructor

```python
EnglishTranscriber(model_name="base", device=None, download_root=None)
```

#### Methods

- `transcribe(audio, **kwargs)`: Main transcription method
- `transcribe_file(audio_file, output_file=None, **kwargs)`: Transcribe and optionally save
- `get_model_info()`: Returns model information dictionary
- `get_available_models()` (static): Returns list of available model names

### Convenience Functions

- `create_transcriber(model_name, device, download_root)`: Factory function
- `transcribe_audio(audio_file, model_name, output_file, **kwargs)`: Quick transcription

For more examples and advanced usage, see the `example_usage.py` file.
