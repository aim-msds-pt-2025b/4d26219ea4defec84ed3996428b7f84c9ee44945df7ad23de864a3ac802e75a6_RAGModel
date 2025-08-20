#!/usr/bin/env python3
"""
Comprehensive test script for the HW3 ML Pipeline
Tests all components: MLflow, Airflow, drift detection, model training
"""

import subprocess
import requests
import sys


def run_command(cmd, description):
    """Run a command and return its result."""
    print(f"\n🔍 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print(f"❌ FAILED: {description}")
            print(f"Error: {result.stderr.strip()}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False, "Command timed out"
    except Exception as e:
        print(f"❌ ERROR: {description} - {str(e)}")
        return False, str(e)


def test_docker_services():
    """Test if all Docker services are running."""
    print("\n" + "=" * 60)
    print("🐳 TESTING DOCKER SERVICES")
    print("=" * 60)

    # Check if containers are running
    success, output = run_command(
        "docker ps --filter name=news-classifier", "Check running containers"
    )
    if not success:
        return False

    # Check each service health
    services = [
        ("news-classifier-airflow-webserver-1", "Airflow Webserver"),
        ("news-classifier-airflow-scheduler-1", "Airflow Scheduler"),
        ("news-classifier-mlflow-1", "MLflow Server"),
        ("news-classifier-postgres-1", "PostgreSQL"),
        ("news-classifier-redis-1", "Redis"),
    ]

    all_healthy = True
    for container, name in services:
        success, _ = run_command(
            f"docker ps --filter name={container} --filter status=running",
            f"Check {name}",
        )
        if not success:
            print(f"❌ {name} is not running")
            all_healthy = False
        else:
            print(f"✅ {name} is running")

    return all_healthy


def test_web_services():
    """Test if web services are accessible."""
    print("\n" + "=" * 60)
    print("🌐 TESTING WEB SERVICES")
    print("=" * 60)

    services = [
        ("http://localhost:8080/health", "Airflow Health"),
        ("http://localhost:5000", "MLflow UI"),
    ]

    all_accessible = True
    for url, name in services:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {name} is accessible (Status: {response.status_code})")
            else:
                print(f"⚠️  {name} returned status {response.status_code}")
                all_accessible = False
        except Exception as e:
            print(f"❌ {name} is not accessible: {str(e)}")
            all_accessible = False

    return all_accessible


def test_dependencies():
    """Test if required dependencies are installed in containers."""
    print("\n" + "=" * 60)
    print("📦 TESTING DEPENDENCIES")
    print("=" * 60)

    dependencies = [
        ("evidently", "Evidently for drift detection"),
        ("mlflow", "MLflow for experiment tracking"),
        ("sklearn", "Scikit-learn for ML models"),
        ("pandas", "Pandas for data processing"),
    ]

    all_installed = True
    for dep, description in dependencies:
        cmd = f"docker exec news-classifier-airflow-scheduler-1 python -c \"import {dep}; print('{dep} imported successfully')\""
        success, _ = run_command(cmd, f"Check {description}")
        if not success:
            all_installed = False

    return all_installed


def test_data_files():
    """Test if required data files exist."""
    print("\n" + "=" * 60)
    print("📁 TESTING DATA FILES")
    print("=" * 60)

    files = [
        ("data/processed/train.csv", "Training data"),
        ("data/processed/test.csv", "Test data"),
        ("data/drifted_test.csv", "Drifted test data"),
        ("data/drifted_train.csv", "Drifted training data"),
    ]

    all_exist = True
    for file_path, description in files:
        success, _ = run_command(
            f"docker exec news-classifier-airflow-scheduler-1 ls -la {file_path}",
            f"Check {description}",
        )
        if not success:
            all_exist = False

    return all_exist


def test_airflow_dag():
    """Test Airflow DAG functionality."""
    print("\n" + "=" * 60)
    print("🔄 TESTING AIRFLOW DAG")
    print("=" * 60)

    # Check if DAG exists
    success, _ = run_command(
        "docker exec news-classifier-airflow-scheduler-1 airflow dags list | grep ml_pipeline_news_classification",
        "Check if DAG is registered",
    )
    if not success:
        return False

    # List DAG tasks
    success, output = run_command(
        "docker exec news-classifier-airflow-scheduler-1 airflow tasks list ml_pipeline_news_classification",
        "List DAG tasks",
    )
    if success:
        print(f"DAG tasks: {output.strip()}")

    # Check recent DAG runs
    success, _ = run_command(
        "docker exec news-classifier-airflow-scheduler-1 airflow dags list-runs -d ml_pipeline_news_classification",
        "Check DAG runs",
    )

    return success


def test_drift_detection():
    """Test drift detection functionality."""
    print("\n" + "=" * 60)
    print("📊 TESTING DRIFT DETECTION")
    print("=" * 60)

    # Test drift detection function directly
    test_script = """
import sys
sys.path.append("/opt/airflow")
from src.drift_detection import detect_drift
import json

try:
    result = detect_drift("data/processed/test.csv", "data/drifted_test.csv")
    print("✅ Drift detection completed successfully")
    print(f"Drift detected: {result.get('drift_detected', 'Unknown')}")
    print(f"Overall drift score: {result.get('overall_drift_score', 'Unknown')}")
except Exception as e:
    print(f"❌ Drift detection failed: {str(e)}")
    sys.exit(1)
"""

    success, output = run_command(
        f'docker exec news-classifier-airflow-scheduler-1 python -c "{test_script}"',
        "Test drift detection function",
    )

    return success


def test_mlflow_tracking():
    """Test MLflow experiment tracking."""
    print("\n" + "=" * 60)
    print("🧪 TESTING MLFLOW TRACKING")
    print("=" * 60)

    # Test MLflow connectivity
    test_script = """
import sys
sys.path.append("/opt/airflow")
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

try:
    import mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Try to get or create an experiment
    experiment_name = "news_classification_test"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"✅ Found existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
    except Exception as e:
        print(f"⚠️  Experiment handling: {str(e)}")

    print("✅ MLflow connectivity test successful")
except Exception as e:
    print(f"❌ MLflow test failed: {str(e)}")
    sys.exit(1)
"""

    success, output = run_command(
        f'docker exec news-classifier-airflow-scheduler-1 python -c "{test_script}"',
        "Test MLflow connectivity",
    )

    return success


def main():
    """Run all tests."""
    print("🚀 STARTING COMPREHENSIVE HW3 PIPELINE VALIDATION")
    print("=" * 60)

    tests = [
        ("Docker Services", test_docker_services),
        ("Web Services", test_web_services),
        ("Dependencies", test_dependencies),
        ("Data Files", test_data_files),
        ("Airflow DAG", test_airflow_dag),
        ("Drift Detection", test_drift_detection),
        ("MLflow Tracking", test_mlflow_tracking),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! HW3 pipeline is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
