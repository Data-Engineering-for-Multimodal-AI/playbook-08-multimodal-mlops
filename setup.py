from setuptools import setup, find_packages

setup(
    name="mlops-pipeline",
    version="0.1.0",
    description="MLOps pipeline for Inertial Navigation System data",
    author="MLOps Team",
    author_email="team@example.com",
    packages=find_packages(include=["pipeline", "pipeline.*", "enhancements_modules", "enhancements_modules.*"]),
    install_requires=[],
    python_requires=">=3.8",
) 