from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS_Hotel_Reservation_Prediction",
    version="0.1",
    description="A package for predicting hotel reservation cancellations",
    author="Hafis Davies",
    packages=find_packages(),
    install_requires=requirements,
)