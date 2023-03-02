# Udacity Data Science Nanodegree: Personal Notes

These are my personal notes taken while following the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

The Nanodegree asssumes basic data analysis skills with python libraries (pandas, numpy, matplotlib, sklearn, etc.) and has 5 modules that build up on those skills; each module has its corresponding folder in this repository with its guide Markdown file:

1. Introduction to Data Science: [`01_Intro_Data_Science`](./01_Intro_Data_Science/DSND_Introduction.md).
2. Software Engineering: [`02_SoftwareEngineering`](./02_SoftwareEngineering/DSND_SWEngineering.md).
3. Data Engineering: [`03_DataEngineering`](./03_DataEngineering/DSND_DataEngineering.md).
4. Experimental Design & Recommendations
5. Data Scientist Capstone

Additionally, it is necessary to submit and pass some projects to get the certification:

- Create a data sciece project and write a blog post: [airbnb_data_analysis](https://github.com/mxagar/airbnb_data_analysis).
- Disaster response pipeline
- Recommendation engine
- Capstone project

A regular python environment with the usual data science packages should suffice (i.e., scikit-learn, pandas, matplotlib, etc.); any special/additional packages and their installation commands are introduced in the guides. A recipe to set up a [conda](https://docs.conda.io/en/latest/) environment with my current packages is the following:

```bash
conda create --name ds pip python=3.10
conda activate ds
pip install -r requirements.txt
```

Mikel Sagardia, 2022.  
No guarantees.


