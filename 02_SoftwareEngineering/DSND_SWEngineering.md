# Udacity Data Science Nanodegree: Introduction

These are my personal notes taken while following the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

The Nanodegree asssumes basic data analysis skills with python libraries (pandas, numpy, matplotlib, sklearn, etc.) and has 5 modules that build up on those skills:

1. Introduction to Data Science
2. Software Engineering
3. Data Engineering
4. Experimental Design & Recommendations
5. Data Scientist Capstone

This folder & guide refer to the **second module**: Software Engineering.

Mikel Sagardia, 2022.
No guarantees.

Overview of Contents:

1. [Introduction to Software Engineering](#1.-Introduction-to-Software-Engineering)
2. [Software Engineering Pratices Part 1](#2.-Software-Engineering-Pratices-Part-1)
3. [Software Engineering Pratices Part 2](#3.-Software-Engineering-Pratices-Part-2)
4. [Introduction to Object Oriented Programming & Python Packages](#4.-Introduction-to-Object-Oriented-Programming-&-Python-Packages)
	- [4.1 Procedural vs. Object Oriented Programming](4.1-Procedural-vs.-Object-Oriented-Programming)
	- [4.2 OOP Syntax in Python](#4.2-OOP-Syntax-in-Python)
		- [Getters & Setters](#Getters-&-Setters)
	- [4.3 Commenting Object-Oriented Code](#4.3-Commenting-Object-Oriented-Code)
5. Portfolio Exercise: Upload a Package to PyPi
6. Web Development
7. Portfolio Exercise: Deploy a Data Dashboard

## 1. Introduction to Software Engineering

This module has 3 major parts and 2 projects that are not compulsory.

Parts:

1. Software Engineering
2. Object Oriented Programming
3. Web Development

Projects:

1. PyPi Package
2. Data Dashboard

The first part is 1:1 the first module of the [Udacity Nanodegree Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821). My notes and practice exercises can be found in this repository:

[mlops_udacity](https://github.com/mxagar/mlops_udacity) `/ 01_Clean_Code`

In order to be up to date with the Web Development requirements, I also watched and made notes on the following course:

[Intro to HTML and CSS](https://learn.udacity.com/courses/ud001)

My notes can be found in my Jekyll guide repo:

[jekyll_web_guide](https://github.com/mxagar/jekyll_web_guide) `/ html_css_bootstrap_guide.md`

The notes are linked on my local computer in the upper folder.

## 2. Software Engineering Pratices Part 1

This section is fully covered in the [Udacity Nanodegree Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821). My notes and practice exercises can be found in this repository:

[mlops_udacity](https://github.com/mxagar/mlops_udacity) `/ 01_Clean_Code`

Topics:

- Clean, efficient and modular code
- Documentation: Code & READMEs
- Git Version Control: Working with Teams

## 3. Software Engineering Pratices Part 2

This section is fully covered in the [Udacity Nanodegree Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821). My notes and practice exercises can be found in this repository:

[mlops_udacity](https://github.com/mxagar/mlops_udacity) `/ 01_Clean_Code`

Topics:

- Testing with `pytest`
- Logging
- Code reviews

## 4. Introduction to Object Oriented Programming & Python Packages

This section has two parts:

1. Object Oriented Programming (OOP)
2. How to build python packages

OOP brings two important advantages:

1. We can write modular programs that can scale more easily.
2. The implementation can be hidden to the user, so that they focus on the functionality.

The ultimate practical goal of the section is to build a Gaussian distirbution class which will be set up as a package and uploaded to PyPi.

Even though the link to the repository with examples provided by Udacity is broken, I found and forked it: [udacity-cd0171--software-engineering-for-data-scientists](https://github.com/mxagar/udacity-cd0171--software-engineering-for-data-scientists).

### 4.1 Procedural vs. Object Oriented Programming

In procedural programming we have variables and functions, which either receive those variables or define new ones in them.

In Object Oriented Programming (OOP), code is **encapsulated** in classes, which are **instantiated** in objects. Classes are like generic blueprints that have:

- Characteristics: **attributes** = variables or other complex objects.
- Actions: **methods** = functions that perform procedures expected from the object.

![OOP: Attributes and Methods = Characteristics and Actions](./pics/OOP_idea.png)


### 4.2 OOP Syntax in Python

```python
# Class definition.
# Always capitalize class names!
# Note: __init__(), self
class Shirt:
	def __init__(self, shirt_color, shirt_size, shirt_size, shirt_price):
		self.color=  shirt_color
		self.size =  shirt_size
		self.style = shirt_style
		self.price = shirt_price

	def change_price(self, new_price):
		self.price = new_price

	def discount(self, discount):
		return self.price * (1-discount)

# Object instantiation: __init__() is the constructor
Shirt('red', 'S', 'short sleeve', 15)

# Object instatiation: object is stored in a variable
new_shirt = Shirt('red', 'S', 'short sleeve', 15)

# We can access the attributes of the objects
print(new_shirt.color)
print(new_shirt.size)
print(new_shirt.style)
print(new_shirt.price)

# Use methods of the object/class
new_shirt.change_price(10)
print(new_shirt.price
print(new_shirt.discount_price(.2))

# Another object
shirt_two = Shirt('orange', 'L', 'short-sleeve', 10)

# Working with two objects
total = shirt_one.price + shirt_two.price
total_discount =  shirt_one.discount(.14) + shirt_two.discount(.06) 

# We can build arrays or other structures that contain objects
tshirt_collection = []
shirt_one = Shirt('orange', 'M', 'short sleeve', 25)
shirt_two = Shirt('red', 'S', 'short sleeve', 15)
shirt_three = Shirt('purple', 'XL', 'short sleeve', 10)
tshirt_collecetion.append(shirt_one)
tshirt_collecetion.append(shirt_two)
tshirt_collecetion.append(shirt_three)

for i in range(tshirt_collection)):
	print (tshirt_collection[i].color)
```

Important notes:

- Capitalize class names.
- `__init__()` is used as a constructor, when classes are instantiated.
- `self` is like a dictionary that holds all attributes & methods and makes them available throughout the class; but all functions need to receive it as the first argument if they want to access the attributes. Behind the scenes, `self` also contains the memory address where the object is located; thus, when a method of a class is called in an instantiated object, the class knows which specific values to catch through `self`.
- Classes are usually defined in python modules/scripts, e.g., `shirt.py`, and we import them: `from shirt import Shirt`.

#### Getters & Setters

In python, the class/object arributes are public: they can be accessed directly: `shirt_one.price`. However, we should write setter and getter functions to access them. Examples why this matters:

1. If we want to get/set the price, the user should not care about any conversions to be made, e.g., EUR <-> USD; now, we might internally modify how prices are stored (EUR/USD), so any conversion should be done by `get_price()` or `set_price()`.
2. What if the attribute is a container of some type (e.g., a set) but we want to return it as another type (e.g., a list). That conversion should be hidden from the user.

Even though all attributes are always public in python, a widespread convenction is to prefix an attribute with `_` when the programmer wants the user to use the associated getters/setters:

```python
# Price is preceded by _ to denote we should use
# its associated getters/setters
class Shirt:
    def __init__(self, shirt_color, shirt_size, shirt_style, shirt_price):
        self._price = shirt_price
		self.color=  shirt_color
		self.size =  shirt_size
		self.style = shirt_style

    def get_price(self):
    	return self._price

    def set_price(self, new_price):
    	self._price = new_price

# However, _price is still accessible
shirt_one = Shirt('yellow', 'M', 'long-sleeve', 15)
shirt_one._price = 12
```

### 4.3 Commenting Object-Oriented Code

Interesting link on how to write docstrings: [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

Notes:

- Document modules, functions and classes.
- Mind the indentation; e.g., in classes, indent the class docstring.
- In functions: args & returns; additionally, write types.

Follow this blueprint:

```python
class Pants:
    """The Pants class represents an article of clothing sold in a store
    """

    def __init__(self, color, waist_size, length, price):
        """Method for initializing a Pants object

        Args: 
            color (str)
            waist_size (int)
            length (int)
            price (float)

        Attributes:
            color (str): color of a pants object
            waist_size (str): waist size of a pants object
            length (str): length of a pants object
            price (float): price of a pants object
        """

        self.color = color
        self.waist_size = waist_size
        self.length = length
        self.price = price

    def change_price(self, new_price):
        """The change_price method changes the price attribute of a pants object

        Args: 
            new_price (float): the new price of the pants object

        Returns: None

        """
        self.price = new_price

    def discount(self, percentage):
        """The discount method outputs a discounted price of a pants object

        Args:
            percentage (float): a decimal representing the amount to discount

        Returns:
            float: the discounted price
        """
        return self.price * (1 - percentage)
```

### 4.4 Gaussian and Binomial Distirbutions

We want to build a Gaussian distribution class and upload it to PyPi. The package will be able to:

- Read in dataset
- Calculate mean
- Calculate the standard deviation
- Plot histogram
- Plot probability density function
- Add two Gaussian distributions

We need to extend the package to handle binomial distirbutions, too.

#### Gaussian Distribution

Probability density function:

`f(x; m, s) = (1/sqrt(2*pi*s^2)) * exp(-((x-m)^2)/(2*s^2))`

#### Binomial Distribution

`mean = n * p`

> A fair coin has a probability of a positive outcome (heads) p = 0.5. If you flip a coin 20 times, the mean would be 20 * 0.5 = 10; you'd expect to get 10 heads.

`variance = n * p*(1−p)`

Probability mass function: [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution).

`f(k,n,p) = (n!/(k!*(n-k)!)) * p^k * (1-p)^(n-k)`

> Assume that 15% of the population is allergic to cats. If you randomly select 60 people for a medical trial, what is the probability that 7 of those people are allergic to cats?

p = 0.15
n = 60
k = 7 (often called x)

`f = 60!/7!*53! * (0.15)^7 * (0.85)^53 = 0.12`.

### 4.5 A Gaussian Class Implementation

We need to implement the following functions of the `Gaussian` class:

	init
	calculate_mean
	calculate_std_dev
	read_data_file
	pdf
	plot_histogram
	plot_histogram_pdf

Notes:

- In an instantiated class, both attributes and methods of the class object are accessed via `self`.
- Use default argument values.

The final code:

```python
import math
import matplotlib.pyplot as plt

class Gaussian():
    """ Gaussian distribution class for calculating and 
    visualizing a Gaussian distribution.
    
    Attributes:
        mean (float) representing the mean value of the distribution
        stdev (float) representing the standard deviation of the distribution
        data_list (list of floats) a list of floats extracted from the data file
    """
    def __init__(self, mu = 0, sigma = 1):
        self.mean = mu
        self.stdev = sigma
        self.data = []
        self.epsilon = 1e-6

    def calculate_mean(self):    
        """Method to calculate the mean of the data set.
        
        Args: 
            None
        Returns: 
            float: mean of the data set
        """
        #TODO: Calculate the mean of the data set. Remember that the data set is stored in self.data
        # Change the value of the mean attribute to be the mean of the data set
        # Return the mean of the data set           
        mu = 0
        for d in self.data:
            mu += d
        self.mean = mu/len(self.data)
        
        return self.mean

    def calculate_stdev(self, sample=True):
        """Method to calculate the standard deviation of the data set.
        
        Args: 
            sample (bool): whether the data represents a sample or population
        Returns: 
            float: standard deviation of the data set
        """
        # TODO:
        #   Calculate the standard deviation of the data set
        #   
        #   The sample variable determines if the data set contains a sample or a population
        #   If sample = True, this means the data is a sample. 
        #   Keep the value of sample in mind for calculating the standard deviation
        #
        #   Make sure to update self.stdev and return the standard deviation as well    
        self.sample = sample
        if math.fabs(self.mean) < self.epsilon:
            self.calculate_mean()
        sigma = 0
        for d in self.data:
            sigma += math.pow((d-self.mean), 2)
        if self.sample:
            sigma /= len(self.data) - 1
        else:
            sigma /= len(self.data)
        sigma = math.sqrt(sigma)
        self.stdev = sigma

        return self.stdev

    def read_data_file(self, file_name, sample=True):    
        """Method to read in data from a txt file. The txt file should have
        one number (float) per line. The numbers are stored in the data attribute. 
        After reading in the file, the mean and standard deviation are calculated
                
        Args:
            file_name (string): name of a file to read from
        Returns:
            None
        """
        # This code opens a data file and appends the data to a list called data_list
        with open(file_name) as file:
            data_list = []
            line = file.readline()
            while line:
                data_list.append(int(line))
                line = file.readline()
        file.close()
        # TODO: 
        #   Update the self.data attribute with the data_list
        #   Update self.mean with the mean of the data_list. 
        #       You can use the calculate_mean() method with self.calculate_mean()
        #   Update self.stdev with the standard deviation of the data_list. Use the 
        #       calculate_stdev() method.
        self.data = data_list
        self.calculate_mean()
        self.calculate_stdev(sample=sample)
        
    def plot_histogram(self):
        """Method to output a histogram of the instance variable data using 
        matplotlib pyplot library.
        
        Args:
            None
        Returns:
            None
        """
        # TODO: Plot a histogram of the data_list using the matplotlib package.
        #       Be sure to label the x and y axes and also give the chart a title
        # make the plot
        fig = plt.figure()
        fig.hist(self.data, density=False)
        fig.set_title('Histogram of Data')
        fig.set_ylabel('Freq. / Count')
        plt.show()
    
    def pdf(self, x):
        """Probability density function calculator for the gaussian distribution.
        
        Args:
            x (float): point for calculating the probability density function
        Returns:
            float: probability density function output
        """
        # TODO: Calculate the probability density function of the Gaussian distribution
        #       at the value x. You'll need to use self.stdev and self.mean to do the calculation
        # `f(x; m, s) = (1/sqrt(2*pi*s^2)) * exp(-((x-m)^2)/(2*s^2))`
        v = math.pow(self.stdev,2)
        m = 1.0/math.sqrt(2.0*math.pi*v)
        f = m*math.exp(-math.pow((x-self.mean),2)/(2.0*v))

        return f

    def plot_histogram_pdf(self, n_spaces = 50):

        """Method to plot the normalized histogram of the data and a plot of the 
        probability density function along the same range
        
        Args:
            n_spaces (int): number of data points 
        
        Returns:
            list: x values for the pdf plot
            list: y values for the pdf plot
        """
        #TODO: Nothing to do for this method. Try it out and see how it works.
        mu = self.mean
        sigma = self.stdev

        min_range = min(self.data)
        max_range = max(self.data)
        
         # calculates the interval between x values
        interval = 1.0 * (max_range - min_range) / n_spaces

        x = []
        y = []
        
        # calculate the x values to visualize
        for i in range(n_spaces):
            tmp = min_range + interval*i
            x.append(tmp)
            y.append(self.pdf(tmp))

        # make the plots
        fig, axes = plt.subplots(2,sharex=True)
        fig.subplots_adjust(hspace=.5)
        axes[0].hist(self.data, density=True)
        axes[0].set_title('Normed Histogram of Data')
        axes[0].set_ylabel('Density')

        axes[1].plot(x, y)
        axes[1].set_title('Normal Distribution for \n Sample Mean and Sample Standard Deviation')
        axes[0].set_ylabel('Density')
        plt.show()

        return x, y
```

### 4.6 Magic Methods: Application to the Summation of Gaussians

Gaussians can be summed simply by summing their parameters:

	Gaussian_C = Gaussian_A + Gaussian_B
	mean_C = mean_A + mean_B
	var_C = var_A + var_B
	std_C = sqrt(var_C)

Python has special magic methods which allow to overload common operators defined for objects; among others, the `__add__` function implements the `+` operator, which can be used to sum to instantiated classes.

Examples for the `Gaussian` class:

```python
    def __add__(self, other):    
        """Magic method to add together two Gaussian distributions
        
        Args:
            other (Gaussian): Gaussian instance
        Returns:
            Gaussian: Gaussian distribution
        """
        result = Gaussian()
        result.mean = self.mean + other.mean
        result.stdev = math.sqrt(math.pow(self.stdev,2) + math.pow(other.stdev,2))
        
        return result

    def __repr__(self):
        """Magic method to output the characteristics of the Gaussian instance.
        
        Args:
            None
        Returns:
            string: characteristics of the Gaussian
        """
        return f"mean {self.mean}, standard deviation {self.stdev}"
```

We can check all the methods (including magic ones) of a type/class with `dir()`:

	dir(int)
	dir(Gaussian)

There are many magic methods; some of them are listed here:

- `__add__`: `+` operator
- `__repr__`: representation string of the object; invoked, for instance when just object is typed or when we `print()` the object.
- `__str__`: stringify an object; similar to `__repr__`.
- `__ge__`: `>=`.
- `__sub__`: `-`
- ...

More on the topic: [Python Magic Methods](https://www.tutorialsteacher.com/python/magic-methods-in-python).

### 4.7 Inheritance





