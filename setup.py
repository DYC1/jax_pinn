from setuptools import setup, find_packages

setup(
    name="jax_pinn",
    version="1.0.1",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "optax",
        "orbax",
        "matplotlib",
        "tqdm",
    ],
    description="A package for managing GPUs and computation precision in JAX for PINN.",
    # long_description=open('README.md').read(),
    # long_description_content_type="text/markdown",
    author="D YC",
    author_email="dyc1go@outlook.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)