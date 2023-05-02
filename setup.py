import setuptools

with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sobolev_alignment",
    version="1.0.0",
    author="Soufiane Mourragui <soufiane.mourragui@gmail.com>, ",
    author_email="soufiane.mourragui@gmail.com",
    description="SOBOLEV ALIGNMENT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NKI-CCB/sobolev_alignment",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "cython",
        "scipy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "torch==1.11.0",
        "scvi-tools==0.20.0",
        "scanpy",
        "hyperopt",
        "codecov",
    ],
    python_requires=">=3.8",
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 1 - Planning",
    ),
)
