
<!-- [![PyPI version](https://badge.fury.io/py/sentimentanalyser.svg)](https://badge.fury.io/py/sentimentanalyser)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HitCount](http://hits.dwyl.io/ashhadulislam/sentiment-analyser-lib.svg)](http://hits.dwyl.io/ashhadulislam/sentiment-analyser-lib)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/sentimentanalyser.svg)](https://img.shields.io/pypi/dm/sentimentanalyser.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/ashhadulislam/sentiment-analyser-lib/badge/master)](https://www.codefactor.io/repository/github/ashhadulislam/sentiment-analyser-lib/overview/master) -->
# Augment data library

### About
A generic package to help data scientists balance their dataset by increasing the datapoints for an imbalanced class.

### Installation

Use below command to install 

`pip install sentimentanalyser`

### Examples

The folder examples contains example application of the data augmentation algorithm for different datasets viz.

* ecoli
* india_liver
* pima_diabetes
* qcri_diabetes
* thyroid

Each of the jupyter notebooks run independedntly and only need the data csv file associated with the code which are present in the data file of the corresponding folder.

### Source

The folder augmentdata contains the source code.



### Usage


Convert your dataset to numpy array.

All values of the data must be numeric.

The last column must be the class label

Function call: 5 inputs
```
augment(data=df.values,k=k,class_ind=1,N=45000,randmx=randmx)
```
- data is the array like input of data, last column of data is class label	
- k is number of neighbors, it should be bigger or equal to 1
- class_ind is the value of data that needs to be augmented. For example, if the class labels are 0 or 1 and the datapoints for 0 need to be upsampled, class_ind=0
- N is the number of Datapoints that needs to be added
- randmx will be a value between 0 and 1, inclusive. smaller the randmx, closer is the data to each original data. randmx, uniform[0,randmx], ; randmx<=1

The outputs are:

- Data_a: complete data with augmented datapoints

- Ext_d: Only the augmented data points

- Ext_not: The datapoints that was created but ignored

Example implementation
```
from augmentdata import data_augment

l=[
[1,3,4,1],
[2,3,4,1],
[1,2,1,0],
[3,2,1,0],
[3,1,1,0],
[2,1,1,0],
[3,2,2,1],
[3,4,2,1],
[4,3,1,1]
]

l=np.array(l)

k=2
randmx=1
daug = data_augment.DataAugment()
[Data_a,Ext_d,Ext_not]=daug.augment(data=l,k=k,class_ind=0,N=5,randmx=randmx)

print(Data_a)
```
Output
```
array([[1.        , 2.        , 1.        , 0.        ],
       [3.        , 2.        , 1.        , 0.        ],
       [3.        , 1.        , 1.        , 0.        ],
       [2.        , 1.        , 1.        , 0.        ],
       [1.29027148, 1.98510073, 1.        , 0.        ],
       [1.65549291, 1.4418645 , 1.        , 0.        ],
       [2.02559196, 1.01965248, 1.        , 0.        ],
       [2.79469135, 1.69371064, 1.        , 0.        ],
       [2.907707  , 1.38716444, 1.        , 0.        ],
       [1.        , 3.        , 4.        , 1.        ],
       [2.        , 3.        , 4.        , 1.        ],
       [3.        , 2.        , 2.        , 1.        ],
       [3.        , 4.        , 2.        , 1.        ],
       [4.        , 3.        , 1.        , 1.        ]])
```

### Read the docs
The documentation of the library is present at

https://augmentdatalib-docs.readthedocs.io/en/latest/

### Citation
If you are using this library in your research please cite the project.


### Authors
- Dr Samir Brahim Belhaouari: samir.brahim@gmail.com, sbelhaouari@hbku.edu.qa
- Ashhadul Islam: ashhadulislam@gmail.com, aislam@mail.hbku.edu.qa
