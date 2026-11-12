from setuptools import setup, find_packages

setup(
    name="hindsight",
    version="0.1.0",
    description="Hindsight: VLM-based Egocentric Object-Memory System with Calibrated Recall",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pillow",
        "pydantic",
    ],
)
