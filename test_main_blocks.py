"""
Simple test script to validate __main__ blocks work correctly.
"""

import subprocess
import sys
import os


def test_main_blocks():
    """Test that all __main__ blocks execute without errors."""

    # Test data_preprocessing.py main block
    print("Testing data_preprocessing.py __main__ block...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.data_preprocessing import *; "
                "import tempfile, pandas as pd; "
                "with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f: "
                "    pd.DataFrame({'text': ['test']*10, 'label': [0]*10}).to_csv(f.name, index=False); "
                "    print('Data preprocessing main block test passed')",
            ],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        print(f"data_preprocessing.py exit code: {result.returncode}")
        if result.stdout:
            print(f"stdout: {result.stdout}")
        if result.stderr:
            print(f"stderr: {result.stderr}")
    except Exception as e:
        print(f"Error testing data_preprocessing.py: {e}")

    # Test download_data.py main block
    print("\nTesting download_data.py __main__ block...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.download_data import *; "
                "print('Download data main block test passed')",
            ],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        print(f"download_data.py exit code: {result.returncode}")
        if result.stdout:
            print(f"stdout: {result.stdout}")
        if result.stderr:
            print(f"stderr: {result.stderr}")
    except Exception as e:
        print(f"Error testing download_data.py: {e}")

    # Test evaluation.py main block
    print("\nTesting evaluation.py __main__ block...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.evaluation import *; "
                "print('Evaluation main block test passed')",
            ],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        print(f"evaluation.py exit code: {result.returncode}")
        if result.stdout:
            print(f"stdout: {result.stdout}")
        if result.stderr:
            print(f"stderr: {result.stderr}")
    except Exception as e:
        print(f"Error testing evaluation.py: {e}")


if __name__ == "__main__":
    test_main_blocks()
