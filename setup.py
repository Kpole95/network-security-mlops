"""
This file is an essential part of packaging and
distributing Python projects. It is used by setuptools
to define the configuration of thr project, such as its
metadata, depencied and more
"""


from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    """
    This funcntion will return the list of requirements
    """

    requirement_lst:List[str]=[]

    try:
        with open('requirements.txt','r') as file:
            # read lines from the file
            lines=file.readlines()

            # process each line
            for line in lines:
                requirement=line.strip() # remove empty spaces

                # ignore the empty line sand -e .
                if requirement and requirement!= '-e .': # not consider -e . in requirements
                    requirement_lst.append(requirement)
    
    except FileNotFoundError:
        print("requiremnts.txt file not found")

    return requirement_lst

print(get_requirements())


setup(
    name="network_security_mlops",
    version="0.0.1",
    author="Krishna Pole",
    author_email="krishnapole90@gmail.com",
    packages=find_packages(),
    install_requirements=get_requirements()    


)