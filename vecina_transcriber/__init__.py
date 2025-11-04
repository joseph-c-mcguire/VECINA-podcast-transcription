"""
VECINA Podcast Transcription Tool

A comprehensive tool for transcribing podcast audio using OpenAI's Whisper model.
Supports local processing and scalable cloud deployment via Modal.
"""

from .transcriber import (
    EnglishTranscriber,
    create_transcriber,
    transcribe_audio,
)

# Version information
__version__ = "0.1.0"
__author__ = "VECINA Team"
__email__ = "contact@vecina.org"

# Package metadata
__title__ = "vecina-transcriber"
__description__ = "A podcast transcription tool using Whisper"
__url__ = "https://github.com/joseph-c-mcguire/VECINA-podcast-transcription"
__license__ = "MIT"
__copyright__ = "2024 VECINA Team"

# Main exports
__all__ = [
    # Core transcription classes and functions
    "EnglishTranscriber",
    "create_transcriber",
    "transcribe_audio",

    # Package metadata
    "__version__",
    "__author__",
    "__title__",
    "__description__",
    "__url__",
    "__license__",
]

# Optional modal imports (only if modal is installed)
try:
    from .modal_entrypoint import (
        transcribe_with_modal,
        batch_transcribe_with_modal,
    )
    __all__.extend([
        "transcribe_with_modal",
        "batch_transcribe_with_modal",
    ])
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False


def is_modal_available() -> bool:
    """Check if Modal deployment functionality is available."""
    return _MODAL_AVAILABLE


def get_version() -> str:
    """Get the package version."""
    return __version__


def get_package_info() -> dict:
    """Get comprehensive package information."""
    return {
        "name": __title__,
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "url": __url__,
        "license": __license__,
        "modal_available": _MODAL_AVAILABLE,
    }
