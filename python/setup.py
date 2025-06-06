import os
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [r for r in requirements if r and not r.startswith("#")]

setup(
    name="omoknuni-mcts",
    version="0.1.0",
    author="Omoknuni Team",
    description="Quantum-inspired Monte Carlo Tree Search for game AI",
    long_description=open("../docs/PRD.md").read() if os.path.exists("../docs/PRD.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "flake8>=6.1.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)