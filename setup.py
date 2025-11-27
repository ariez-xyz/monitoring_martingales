from setuptools import setup, find_packages

setup(
    name="neural-control-monitoring",
    version="0.1.0",
    packages=find_packages(include=['monitor*', 'sablas*']),
)
