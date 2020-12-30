import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numpy_blocks-amice",
    version="0.0.1",
    author="Alexandre Amice",
    author_email="amice@mit.edu",
    description="A package for conveniently handling block indexing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexandreAmice/numpy_blocks",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license='MIT',
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires = [
        'numpy>=1.16.6',
        'scipy>=1.2.2'
    ]
)