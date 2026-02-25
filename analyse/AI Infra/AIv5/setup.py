from setuptools import setup, find_packages

setup(
    name="aiv5",
    version="5.0.0",
    packages=find_packages(include=[
        'pattern_detection', 'pattern_detection.*',
        'training', 'training.*',
        'data_acquisition', 'data_acquisition.*',
        'shared', 'shared.*',
        'src', 'src.*'
    ]),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "pyarrow>=10.0.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.7.0",
        "lightgbm>=3.3.0",
        "tqdm>=4.60.0",
        "pydantic>=2.0.0",
        "google-cloud-storage>=2.0.0",
        "yfinance>=0.2.0",
    ],
    python_requires=">=3.8",
    description="AIv5 Pattern Detection and ML Training System - Clean Production Version",
)
