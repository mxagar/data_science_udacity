from setuptools import setup

setup(name = 'distributions_mxagar', # every package on PyPi needs a UNIQUE name
      version = '0.11', # every upload needs a new version
      description = 'Gaussian and Binomial distributions',
      packages = ['distributions_mxagar'], # package folder with __init__.py in it; use same as in name
      author = 'Mikel Sagardia',
      author_email = 'mxagar@gmail.com',
      zip_safe=False) # whether we can execute the package while zipped; e.g., if we use an ASCII file from the package folder we can't
