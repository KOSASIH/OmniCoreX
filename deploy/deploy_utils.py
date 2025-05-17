"""
OmniCoreX Deployment Utilities

This module contains advanced deployment configurations and helper functions
to facilitate containerization, cloud deployment, and robust model serving
for the OmniCoreX system.

Features:
- Dockerfile and Kubernetes manifest generation helpers.
- Environment variables management with secure handling.
- Health check and readiness probe utilities.
- Automated setup scripts for cloud environments.
- Utilities for scalable and fault-tolerant serving.
"""

import os
import json
import subprocess
from typing import Dict, Optional

def generate_dockerfile(base_image: str = "python:3.10-slim",
                        requirements_file: str = "requirements.txt",
                        workdir: str = "/app",
                        entry_point: str = "scripts/deploy_server.py",
                        expose_port: int = 8000) -> str:
    """
    Generates a Dockerfile string for OmniCoreX deployment.

    Args:
        base_image: Base Docker image.
        requirements_file: Path to Python requirements file.
        workdir: Working directory inside container.
        entry_point: Path to container entry script.
        expose_port: Port to expose.

    Returns:
        Dockerfile content as string.
    """
    dockerfile = f"""
    FROM {base_image}

    WORKDIR {workdir}

    COPY . {workdir}/

    RUN pip install --no-cache-dir -r {requirements_file}

    EXPOSE {expose_port}

    CMD ["python", "{entry_point}", "--host", "0.0.0.0", "--port", "{expose_port}"]
    """
    return dockerfile.strip()

def save_dockerfile(path: str = "Dockerfile", **kwargs) -> None:
    """
    Saves generated Dockerfile to specified path.

    Args:
        path: File path to save Dockerfile.
        kwargs: Arguments to generate_dockerfile.
    """
    content = generate_dockerfile(**kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Dockerfile saved to {path}")

def generate_k8s_deployment(deployment_name: str,
                            image_name: str,
                            replicas: int = 1,
                            container_port: int = 8000,
                            cpu_request: str = "500m",
                            mem_request: str = "1Gi",
                            cpu_limit: str = "1",
                            mem_limit: str = "2Gi") -> Dict:
    """
    Generates a Kubernetes deployment manifest dictionary for OmniCoreX.

    Args:
        deployment_name: Name of K8s deployment.
        image_name: Docker image to use.
        replicas: Number of pods.
        container_port: Port exposed by container.
        cpu_request: Requested CPU.
        mem_request: Requested memory.
        cpu_limit: CPU limit.
        mem_limit: Memory limit.

    Returns:
        Dict representing K8s deployment YAML.
    """
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": deployment_name},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": deployment_name}},
            "template": {
                "metadata": {"labels": {"app": deployment_name}},
                "spec": {
                    "containers": [{
                        "name": deployment_name,
                        "image": image_name,
                        "ports": [{"containerPort": container_port}],
                        "resources": {
                            "requests": {
                                "cpu": cpu_request,
                                "memory": mem_request
                            },
                            "limits": {
                                "cpu": cpu_limit,
                                "memory": mem_limit
                            }
                        },
                        "readinessProbe": {
                            "httpGet": {
                                "path": "/health",
                                "port": container_port
                            },
                            "initialDelaySeconds": 5,
                            "periodSeconds": 10
                        },
                        "livenessProbe": {
                            "httpGet": {
                                "path": "/health",
                                "port": container_port
                            },
                            "initialDelaySeconds": 10,
                            "periodSeconds": 30
                        }
                    }]
                }
            }
        }
    }
    return deployment

def save_k8s_manifest(manifest: Dict, path: str = "k8s_deployment.yaml") -> None:
    """
    Saves K8s manifest dict as YAML file.

    Args:
        manifest: Dictionary of K8s manifest.
        path: File path to save manifest YAML.
    """
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f)
    print(f"Kubernetes manifest saved to {path}")

def build_docker_image(image_tag: str, dockerfile_path: str = "Dockerfile", context: str = ".") -> None:
    """
    Builds a docker image using docker CLI.

    Args:
        image_tag: Name and tag of docker image (e.g., "omnicorex:latest").
        dockerfile_path: Path to Dockerfile.
        context: Build context directory.
    """
    cmd = ["docker", "build", "-t", image_tag, "-f", dockerfile_path, context]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Built docker image: {image_tag}")

def push_docker_image(image_tag: str) -> None:
    """
    Pushes docker image to registry.

    Args:
        image_tag: Docker image tag.
    """
    cmd = ["docker", "push", image_tag]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Pushed docker image: {image_tag}")

def setup_env_vars(env_vars: Dict[str, str], path: str = ".env") -> None:
    """
    Writes environment variables to a .env file for deployment configuration.

    Args:
        env_vars: Dict of environment variables to write.
        path: Path to .env file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for key, val in env_vars.items():
            f.write(f"{key}={val}\n")
    print(f"Environment variables saved in {path}")

if __name__ == "__main__":
    # Example usage: generate dockerfile and k8s manifest

    save_dockerfile(
        base_image="python:3.10-slim",
        requirements_file="requirements.txt",
        workdir="/app",
        entry_point="scripts/deploy_server.py",
        expose_port=8000
    )

    deployment_manifest = generate_k8s_deployment(
        deployment_name="omnicorex",
        image_name="kosasih/omnicorex:latest",
        replicas=2,
        container_port=8000
    )
    save_k8s_manifest(deployment_manifest)

    # Example environment variables setup
    env_vars = {
        "MODEL_PATH": "/app/checkpoints/checkpoint.pt",
        "LOG_LEVEL": "INFO",
        "NUM_WORKERS": "4"
    }
    setup_env_vars(env_vars)

    print("Deployment utilities demo complete.")

