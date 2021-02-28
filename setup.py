from setuptools import setup, find_packages
# Read requirements.txt, ignore comments
#try:
#    REQUIRES = list()
#    f = open("requirements.txt", "rb")
#    for line in f.read().decode("utf-8").split("\n"):
#        line = line.strip()
#        if "#" in line:
#            line = line[: line.find("#")].strip()
#        if line:
#            REQUIRES.append(line)
    # print(REQUIRES)
#except:
#    print("'requirements.txt' not found!")
#    REQUIRES = list()
setup(
    name="elegantrl",
    version="0.0.1",
    #include_package_data=True,
    author="author",
    author_email="author_email",
    url="https://github.com/AI4Finance-LLC/ElegantRL",
    license="MIT",
    #packages=find_packages(),
    # install_requires=REQUIRES,
    #install_requires=REQUIRES,
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
