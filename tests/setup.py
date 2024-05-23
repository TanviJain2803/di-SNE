from setuptools import setup

setup(
    name='diSNE',
    version='1.0',
    description='CSE 185 Project',
    author='Tanvi Jain, Nimrit Kaur, Kathryn Chen',
    author_email='kac012@ucsd.edu',
    packages=['diSNE'],
    entry_points={
        "console_scripts": [
              "diSNE=diSNE.diSNE:main"
        ],
    },
)
