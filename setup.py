from setuptools import setup, find_packages
# List of requirements
requirements = []  # This could be retrieved from requirements.txt
# Package (minimal) configuration
setup(
    name="wipy",
    version="0.0.1",
    description="waveform inversion in python",
    packages=find_packages(),  # __init__.py folders search
    install_requires=requirements
)
