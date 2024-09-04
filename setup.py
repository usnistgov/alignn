"""ALIGNN: Atomistic LIne Graph Neural Network.

https://jarvis.nist.gov.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alignn",
    version="2024.8.27",
    author="Kamal Choudhary, Brian DeCost",
    author_email="kamal.choudhary@nist.gov",
    description="alignn",
    install_requires=[
        "numpy>=1.19.5,<2.0.0",
        "scipy>=1.6.1",
        "jarvis-tools>=2021.07.19",
        "torch>=2.0.0",
        "mpmath<=1.3.0",
        "dgl>=0.6.0",
        "spglib>=2.0.2",
        "scikit-learn>=0.22.2",
        "matplotlib>=3.4.1",
        "tqdm>=4.60.0",
        "pandas>=1.2.3",
        "pydantic>=1.8.1",
        "pydantic-settings",
        "flake8>=3.9.1",
        "pycodestyle>=2.7.0",
        "pydocstyle>=6.0.0",
        "pyparsing>=2.2.1,<3",
        "ase",
        "lmdb",
        "train_on_folder",
        # "pytorch-ignite>=0.5.0.dev20221024",
        # "accelerate>=0.20.3",
        # "dgl-cu101>=0.6.0",
    ],
    scripts=[
        "alignn/pretrained.py",
        "alignn/train_alignn.py",
        "alignn/run_alignn_ff.py",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usnistgov/alignn",
    packages=setuptools.find_packages(),
    package_data={"alignn.ff": ["all_models_ff.json"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
