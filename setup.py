from setuptools import setup, find_packages

setup(
    name="MLOps_Enhancements",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=1.8.0",
        "pyarrow>=7.0.0",
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0"
    ],
) 