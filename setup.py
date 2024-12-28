from setuptools import setup, find_packages

setup(
    name="superpoint_transformer",
    version="0.1",
    packages=find_packages(where="src"),  # if your code is under src/
    package_dir={"": "src"},              # use 'src' as the root for packages
    install_requires=[],                  # add any dependencies if needed
)
