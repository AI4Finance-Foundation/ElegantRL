from setuptools import setup, find_packages

setup(
    name="ElegantRL",
    version="0.3.7",
    author="AI4Finance Foundation",
    author_email="contact@ai4finance.org",
    url="https://github.com/AI4Finance-Foundation/ElegantRL",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        "th",
        "numpy",
        "matplotlib",
        "gym",
        "gym[Box2D]",
    ],
    description="Lightweight, Efficient and Stable DRL Implementation Using PyTorch",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Deep Reinforcement Learning",
    python_requires=">=3.6",
)
