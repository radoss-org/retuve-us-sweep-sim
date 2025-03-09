from setuptools import find_packages, setup

# Read the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="us_sweep_sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
)
