from setuptools import setup, find_packages

setup(
    name="elegantrl",
    version="0.3.0",
    author="Xiaoyang Liu, Steven Li, Hongyang Yang, Jiahao Zheng",
    author_email="XL2427@columbia.edu",
    url="https://github.com/AI4Finance-LLC/ElegantRL",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        'gym', 'matplotlib', 'numpy', 'pybullet', 'torch'],
    description="Lightweight, Efficient and Stable DRL Implementation Using PyTorch",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Deep Reinforcment Learning",
    python_requires=">=3.6",
)
