# Recommender System: Package

This folder contains the scripts of the recommender system implemented in the notebooks.

Even though it is not really a package (`__init__.py` & Co. are missing), it is the first step to create one.

We have 3 files:

- [`recommender_template.py`](recommender_template.py): empty file where the `Recommender` class is to be defined.
- [`recommender.py`](recommender.py): the already implemented `Recommender` class which uses the FunkSVD algorithm to factorize  a user-item matrix, using the usual ML functions, i.e., `fit()`.
- [`recommender_functions.py`](recommender_functions.py): all auxiliary functions used in the notebooks:
    - `get_movie_names()`
    - `create_ranked_df()`
    - `find_similar_movies()`
    - `popular_recommendations()`

To use the recommender, we can either run `recommender.py` (because it has a `__main__`) or in a python session located inside the folder, instantiate a `Recommender` object:

```bash
cd .../package
python recommender.py
# Optimizaiton Statistics
# Iterations | Mean Squared Error 
# 1 		 15.924775
# 2 		 10.295270
# ...
# 50 		 0.100849

# For user 8 we predict a 5.84 rating for the movie  Fantômas - À l'ombre de la guillotine (1913).

# (array([2125608, 1853728, 2401621, 1205489, 1255953]),
# ['Gran Torino (2008)', 'Incendies (2010)', 'Django Unchained (2012)', 'Searching for Sugar Man (2012)', 'De Marathon (2012)'])

# Because this user wasn't in our database, we are giving back the top movie recommendations for all users.
# (None, ['Goodfellas (1990)', 'Step Brothers (2008)', 'American Beauty (1999)', 'There Will Be Blood (2007)', 'Gran Torino (2008)'])

# (None, ['Killers of the Flower Moon (2021)'])

# That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.
# (None, None)

# 3278
# 2679
# 8000
```
