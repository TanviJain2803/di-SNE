setup(
    name='diSNE',
    version=VERSION,
    description='CSE 185 Project',
    author='',
    author_email='',
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "diSNE=di-SNE.diSNE:main"
        ],
    },
)