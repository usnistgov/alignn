"""Jarvisdgl: Deep graph library for materials science.

https://jarvis.nist.gov.
"""

import glob
import os

from setuptools import find_packages, setup

JARVIS_DIR = os.path.dirname(os.path.abspath(__file__))

base_dir = os.path.dirname(__file__)
with open(os.path.join(base_dir, "README.md")) as f:
    long_d = f.read()

setup(
    name="jarvisdgl",
    version="2021.2.22",
    long_description=long_d,
    install_requires=[
        "numpy>=1.18.5",
        "scipy>=1.4.1",
        "matplotlib>=3.0.0",
        "jarvis-tools==2021.2.21",
        "dgl==0.5.3",
        "torch==1.7.1",
        "scikit-learn==0.24.1",
        "pytorch-ignite",
        "pydantic",
        "tqdm",
        "pymatgen",
    ],
    author="Kamal Choudhary, Brian DeCost",
    author_email="kamal.choudhary@nist.gov",
    description=(
        "jarvisdgl: Deep graph library for materials science. https://jarvis.nist.gov/"  # noqa:501
    ),
    license="NIST",
    url="https://github.com/JARVIS-Materials-Design/jarvisdgl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    # scripts=glob.glob(os.path.join(JARVIS_DIR,  "*"))
)
