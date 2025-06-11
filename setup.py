from setuptools import setup, find_packages
from typing import List


# Function for requirements.txt

HYPEN_E_DOT = "-e ."
def get_requirements(filepath:str)->list[str]:
    """
    This function returns the list of requirements
    """
    requirements = []
    with open(filepath) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements





setup(
    name="Machine Learning Project",
    version="0.0.1",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)