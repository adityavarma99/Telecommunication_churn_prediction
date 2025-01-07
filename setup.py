from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT="-e ."

def get_requirements(file_path:str)->List[str]:
    """
    this function will return the list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
         requirements=file_obj.readlines()                               ## we are reading the pandas, numpy lines in reqirements file
         requirements=[req.replace("\n","") for req in requirements]      ## as pandas and numpy as one after another, we will replace next line/n with blank or space

         if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
name='Telecommunication_churn_prediction',
version="0.0.1",
author="Aditya",
author_email="adityavarma759@gmail.com",
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)

