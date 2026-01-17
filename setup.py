from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() \
                if line.strip() and not line.startswith("#")]

setup(
    name="visionforge",
    version="0.1.0",
    packages=find_packages(include=['orchard', 'orchard.*']),
    python_requires=">=3.10",
    install_requires=parse_requirements('requirements.txt'),
)