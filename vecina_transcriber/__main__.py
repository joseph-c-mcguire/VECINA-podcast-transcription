"""
The purpose of this module is to run main function in cli.py.
So that this can be ran as a module using `python -m vecina_transcriber`.

This module serves as the entry point when the package is executed as a module,
providing command-line interface functionality for the VECINA transcription tool.
"""

import sys


def main() -> None:
    """Main entry point for module execution."""
    try:
        from .cli import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"Error importing CLI module: {e}", file=sys.stderr)
        print("Make sure all dependencies are installed:", file=sys.stderr)
        print("  pip install -e .", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except (RuntimeError, OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
