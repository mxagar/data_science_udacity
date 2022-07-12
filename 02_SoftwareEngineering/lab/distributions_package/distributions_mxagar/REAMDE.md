# distributions_mxagar

This folder contains an exemplary python package uploaded to PyPi. Concretely, the package contains a Gaussian distirbution class and a Binomial distirbution class, with which basic operations can be performed

This package is developed in the module **Software Engineering** of the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

The original code comes from the course creators and it can be partially obtained from the following Udacity repository:

[https-github.com-udacity-cd0171--software-engineering-for-data-scientists](https://github.com/udacity/https-github.com-udacity-cd0171--software-engineering-for-data-scientists)

## Contents

- `Generaldistribution.py`: base distribution class from which the two other classes are inherited. It contains a constructor which initializes the distribution parameters (mean and std. dev.) as well a method to read a data file with a sample of numbers to find the parameters based on it.
- `Binomialdistribution.py`: Binomial distribution with basic methods: magic `__add__` and `__repr__`, plotting, computation of parameters from data (mean and std. dev.) as well as the computation of the PDF.
- `Gaussiandistribution.py`: Gausssian distribution with basic methods: magic `__add__` and `__repr__`, plotting, computation of parameters from data (mean and std. dev.) as well as the computation of the PDF.

The rest of the files are the expected ones in a Python package.

## How to Use This

Installation in your preferred virtual environment:

```
pip install distributions-mxagar
```

Usage of the package:

```python
from distributions_mxagar import Gaussian, Binomial

gaussian_one = Gaussian(10,5)
gaussian_one.mean # 10
gaussian_one.stdev # 5

gaussian_two = Gaussian(1,2)
gaussian_three = gaussian_one + gaussian_two

binomial = Binomial() # prob=.5, size=20
binomial # mean 10.0, standard deviation 2.23606797749979, p 0.5, n 20
```