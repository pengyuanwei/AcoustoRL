from setuptools import setup, find_packages

setup(
    name="AcoustoRL",
    version="0.1.0",
    author="Pengyuan Wei",
    author_email="ucabpw3@ucl.ac.uk",
    license='MIT',
    description="A RL implementation Python library",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AcoustoRL",
    packages=find_packages(),
    install_requires=None,    
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)