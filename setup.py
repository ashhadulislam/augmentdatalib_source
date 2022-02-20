import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name='knnor',
      version='0.0.4',
      description='Generic python library to perform augmentation of data',
      long_description='A generic package to help data scientists balance their dataset by increasing the datapoints for an imbalanced class.'
                  ,
      url='',
      author='Ashhadul Islam, Sameer Brahim Belhaouari, Atiq Ur Rahman, Halima Bensmail',
      author_email='ashhadulislam@gmail.com, samir.brahim@gmail.com, atrehman2@hbku.edu.qa, hbensmail@hbku.edu.qa',
      keywords='Binary Classification, Data Augmentation, Imabalnced Data',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[            
            "numpy",
            "scikit-learn"
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False
)