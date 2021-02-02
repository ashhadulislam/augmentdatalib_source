import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name='augmentdata',
      version='0.0.11',
      description='Generic python library to perform augmentation of data',
      long_description='A generic package to help data scientists balance their dataset by increasing the datapoints for an imbalanced class.'
                  ,
      url='',
      author='Sameer Brahim Belhaouari, Ashhadul Islam',
      author_email='samir.brahim@gmail.com, ashhadulislam@gmail.com',
      keywords='Binary Classification, Data Augmentation, Imabalnced Data',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[            
            "numpy"
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False
)