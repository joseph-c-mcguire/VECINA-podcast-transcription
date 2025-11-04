#!/usr/bin/env python3
"""
Test script for the English transcriber module.
"""

from vecina_transcriber.transcriber import EnglishTranscriber, create_transcriber


def test_transcriber_creation():
    """Test transcriber creation and model info."""
    print("Testing transcriber creation...")

    # Test factory function
    transcriber = create_transcriber(model_name="tiny")

    # Get model info
    info = transcriber.get_model_info()
    print(f"Model info: {info}")

    # Get available models
    models = EnglishTranscriber.get_available_models()
    print(f"Available models: {models}")

    print("✓ Transcriber creation test passed!")


def test_basic_functionality():
    """Test basic functionality without actual audio file."""
    print("\nTesting basic functionality...")

    transcriber = create_transcriber(model_name="tiny")

    # Test that the transcriber is properly initialized
    assert transcriber.model is not None, "Model should be loaded"
    assert transcriber.model_name == "tiny", "Model name should match"

    print("✓ Basic functionality test passed!")


if __name__ == "__main__":
    print("Running English Transcriber Tests")
    print("=" * 40)

    try:
        test_transcriber_creation()
        test_basic_functionality()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
