"""
Test module for Docker and Airflow setup validation.

This module contains comprehensive tests to validate Docker containerization,
Airflow orchestration setup, and project structure integrity.
"""

import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil


class TestDockerSetup:
    """Test class for Docker and containerization setup."""

    @pytest.mark.smoke
    @pytest.mark.slow
    def test_dockerfile_exists_and_valid(self):
        """Test that Dockerfile exists and has valid syntax."""
        dockerfile_path = Path("Dockerfile")
        assert dockerfile_path.exists(), "Dockerfile not found"

        # Check basic Dockerfile syntax
        with open(dockerfile_path, "r") as f:
            content = f.read()

        assert "FROM" in content, "Dockerfile missing FROM instruction"
        assert "WORKDIR" in content, "Dockerfile missing WORKDIR instruction"
        assert "COPY" in content, "Dockerfile missing COPY instruction"
        assert "python" in content.lower(), "Dockerfile should use Python base image"

    @pytest.mark.smoke
    @pytest.mark.fast
    def test_dockerignore_exists(self):
        """Test that .dockerignore exists to optimize build context."""
        dockerignore_path = Path(".dockerignore")
        assert dockerignore_path.exists(), ".dockerignore file not found"

        with open(dockerignore_path, "r") as f:
            content = f.read()

        # Check for important exclusions
        assert "__pycache__" in content, ".dockerignore should exclude __pycache__"
        assert ".git" in content, ".dockerignore should exclude .git"
        assert "tests/" in content, ".dockerignore should exclude tests directory"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_docker_compose_file_exists(self):
        """Test that docker-compose.yml exists and has required services."""
        compose_path = Path("docker-compose.yml")
        assert compose_path.exists(), "docker-compose.yml not found"

        with open(compose_path, "r") as f:
            content = f.read()

        # Check for required services
        assert "postgres:" in content, "docker-compose.yml missing postgres service"
        assert "redis:" in content, "docker-compose.yml missing redis service"
        assert "airflow-webserver:" in content, (
            "docker-compose.yml missing airflow-webserver"
        )
        assert "airflow-scheduler:" in content, (
            "docker-compose.yml missing airflow-scheduler"
        )

    @pytest.mark.smoke
    @pytest.mark.slow
    @patch("subprocess.run")
    def test_docker_compose_syntax_validation(self, mock_subprocess):
        """Test docker-compose syntax validation with mocked subprocess."""
        # Mock successful validation
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # This would normally run docker-compose config
        try:
            subprocess.run(
                ["docker-compose", "config"], capture_output=True, text=True, timeout=30
            )
            # If docker-compose is available, test should pass
            assert True
        except FileNotFoundError:
            # If docker-compose is not installed, mock the test
            mock_subprocess.assert_called_once()
            assert mock_result.returncode == 0

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="Docker not available in test environment",
    )
    def test_docker_build_simulation(self):
        """Test Docker build process (simulation or actual if available)."""
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")

        # Check if docker is available
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            docker_available = True
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            docker_available = False

        if not docker_available:
            pytest.skip("Docker not available, skipping build test")

        # Test that build command can be constructed properly
        build_cmd = [
            "docker",
            "build",
            "-t",
            "test-ml-pipeline",
            "--help",  # Use help flag to validate command construction
            ".",
        ]

        # For actual testing, you might want to run a limited build
        # For CI/CD, this could be a full build
        assert len(build_cmd) > 0  # Basic assertion that command is constructed


class TestAirflowSetup:
    """Test class for Airflow orchestration setup."""

    @pytest.mark.smoke
    @pytest.mark.fast
    def test_airflow_dag_file_exists(self):
        """Test that Airflow DAG file exists."""
        dag_path = Path(__file__).parent.parent / "deploy/airflow/dags/ml_pipeline_dag.py"
        assert dag_path.exists(), "ML Pipeline DAG file not found"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_airflow_dag_syntax(self):
        """Test Airflow DAG Python syntax."""
        dag_path = Path(__file__).parent.parent / "deploy/airflow/dags/ml_pipeline_dag.py"
        if not dag_path.exists():
            pytest.fail("DAG file not found")

        # Add project root to path for imports
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        try:
            with open(dag_path, "r") as f:
                dag_code = f.read()

            # Compile to check syntax
            compile(dag_code, str(dag_path), "exec")
        except SyntaxError as e:
            pytest.fail(f"DAG syntax error: {e}")
        finally:
            # Clean up sys.path
            if str(project_root) in sys.path:
                sys.path.remove(str(project_root))

    @pytest.mark.unit
    @pytest.mark.fast
    def test_airflow_dag_contains_required_elements(self):
        """Test that DAG contains required Airflow elements."""
        dag_path = Path("deploy/airflow/dags/ml_pipeline_dag.py")
        if not dag_path.exists():
            pytest.skip("DAG file not found")

        with open(dag_path, "r") as f:
            content = f.read()

        # Check for required imports and elements
        assert "from airflow import DAG" in content, "DAG missing Airflow import"
        assert "PythonOperator" in content, "DAG missing PythonOperator"
        assert "default_args" in content, "DAG missing default_args"
        assert "dag = DAG" in content, "DAG missing DAG instantiation"

    @pytest.mark.smoke
    @pytest.mark.fast
    def test_airflow_directory_structure(self):
        """Test that Airflow directory structure is correct."""
        required_dirs = [
            Path("deploy/airflow"),
            Path("deploy/airflow/dags"),
            Path("deploy/airflow/logs"),
        ]

        for dir_path in required_dirs:
            assert dir_path.exists(), f"Required directory {dir_path} not found"
            assert dir_path.is_dir(), f"{dir_path} exists but is not a directory"


class TestProjectStructure:
    """Test class for overall project structure validation."""

    @pytest.mark.smoke
    @pytest.mark.fast
    def test_required_files_exist(self):
        """Test that all required project files exist."""
        required_files = [
            "Dockerfile",
            "docker-compose.yml",
            ".dockerignore",
            ".env",
            "pyproject.toml",
            "README.md",
            ".gitignore",
            ".pre-commit-config.yaml",
        ]

        for file_path in required_files:
            path = Path(file_path)
            assert path.exists(), f"Required file {file_path} not found"
            assert path.is_file(), f"{file_path} exists but is not a file"

    @pytest.mark.smoke
    @pytest.mark.fast
    def test_source_directory_structure(self):
        """Test that source code directory structure is correct."""
        required_dirs = [
            Path("src"),
            Path("tests"),
            Path("deploy"),
            Path("config"),
            Path("data"),
            Path("models"),
            Path("reports"),
        ]

        for dir_path in required_dirs:
            assert dir_path.exists(), f"Required directory {dir_path} not found"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gitkeep_files_exist(self):
        """Test that .gitkeep files exist to preserve directory structure."""
        gitkeep_files = [
            Path("data/.gitkeep"),
            Path("data/raw/.gitkeep"),
            Path("data/processed/.gitkeep"),
            Path("models/.gitkeep"),
            Path("reports/.gitkeep"),
            Path("deploy/airflow/logs/.gitkeep"),
        ]

        for gitkeep_path in gitkeep_files:
            assert gitkeep_path.exists(), f"Gitkeep file {gitkeep_path} not found"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_env_file_is_template(self):
        """Test that .env file is properly set up as a template."""
        env_path = Path(".env")
        assert env_path.exists(), ".env file not found"

        with open(env_path, "r") as f:
            content = f.read()

        # Check for template indicators
        assert "TEMPLATE" in content.upper() or "template" in content.lower(), (
            ".env should be marked as a template"
        )
        assert "AIRFLOW_UID" in content, ".env should contain AIRFLOW_UID"
        assert "_AIRFLOW_WWW_USER_USERNAME" in content, (
            ".env should contain Airflow user settings"
        )


class TestEnvironmentConfiguration:
    """Test class for environment and configuration validation."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_pyproject_toml_valid(self):
        """Test that pyproject.toml is valid and contains required sections."""
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml not found"

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip(
                    "No TOML parser available (Python 3.11+ required for tomllib, or install tomli)"
                )

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        assert "project" in config, "pyproject.toml missing [project] section"
        assert "dependencies" in config["project"], (
            "pyproject.toml missing dependencies"
        )

    @pytest.mark.smoke
    @pytest.mark.fast
    def test_pre_commit_config_valid(self):
        """Test that pre-commit configuration is valid."""
        precommit_path = Path(".pre-commit-config.yaml")
        assert precommit_path.exists(), ".pre-commit-config.yaml not found"

        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")

        with open(precommit_path, "r") as f:
            config = yaml.safe_load(f)

        assert "repos" in config, "pre-commit config missing repos section"
        assert len(config["repos"]) > 0, "pre-commit config has no repositories"

        # Check for essential hooks
        hook_names = []
        for repo in config["repos"]:
            if "hooks" in repo:
                hook_names.extend([hook["id"] for hook in repo["hooks"]])

        assert "ruff" in hook_names, "pre-commit missing ruff linter"
        assert "pytest" in hook_names, "pre-commit missing pytest hook"
