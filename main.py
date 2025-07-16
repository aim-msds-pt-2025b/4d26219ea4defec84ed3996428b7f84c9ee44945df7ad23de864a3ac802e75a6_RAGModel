"""
Main entry point for the RAG Model project.

This file provides a simple entry point to run the ML pipeline.
The actual implementation is in src/run_pipeline.py.
"""

from src.run_pipeline import main as run_pipeline


def main():
    """Run the complete ML pipeline."""
    print("Starting RAG Model ML Pipeline...")
    try:
        run_pipeline()
        print("ML Pipeline completed successfully!")
    except Exception as e:
        print(f"ML Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
