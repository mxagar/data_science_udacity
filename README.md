# Udacity Data Science Nanodegree: Personal Notes

These are my personal notes taken while following the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

The Nanodegree assumes basic data analysis skills with python libraries (pandas, numpy, matplotlib, sklearn, etc.) and has 5 modules that build up on those skills; each module has its corresponding folder in this repository with its guide Markdown file:

1. Introduction to Data Science: [`01_Intro_Data_Science`](./01_Intro_Data_Science/DSND_Introduction.md).
2. Software Engineering: [`02_SoftwareEngineering`](./02_SoftwareEngineering/DSND_SWEngineering.md).
3. Data Engineering: [`03_DataEngineering`](./03_DataEngineering/DSND_DataEngineering.md).
4. Experimental Design & Recommendations: [`04_ExperimentalDesign_RecSys`](./04_ExperimentalDesign_RecSys/DSND_ExperimentalDesign_RecSys.md).
5. Data Scientist Capstone (Spark): [`05_Capstone_Project`](./05_Capstone_Project/DSND_Capstone.md).

Additionally, it is necessary to submit and pass some projects to get the certification:

- Create a data science project and write a blog post: [airbnb_data_analysis](https://github.com/mxagar/airbnb_data_analysis).
- Disaster response prediction pipeline deployed on a Flask app: [disaster_response_pipeline](https://github.com/mxagar/disaster_response_pipeline).
- Recommender System which suggests new articles to the users of the IBM Watson Studio Platform: [recommendations_ibm](https://github.com/mxagar/recommendations_ibm).
- Capstone project: [Spark Big Data Guide](https://github.com/mxagar/spark_big_data_guide).

A regular python environment with the usual data science packages should suffice (i.e., scikit-learn, pandas, matplotlib, etc.); any special/additional packages and their installation commands are introduced in the guides. A recipe to set up a [conda](https://docs.conda.io/en/latest/) environment with my current packages is the following:

```bash
conda create --name ds pip python=3.10
conda activate ds
pip install -r requirements.txt
```

As a side note, I list here some related **free Udacity courses** on several topics:

- **Big Data**
  - [Intro to Hadoop and MapReduce](https://www.udacity.com/course/intro-to-hadoop-and-mapreduce--ud617)
  - [Deploying a Hadoop Cluster](https://www.udacity.com/course/deploying-a-hadoop-cluster--ud1000)
  - [Real-Time Analytics with Apache Storm](https://www.udacity.com/course/real-time-analytics-with-apache-storm--ud381)
  - [Big Data Analytics in Healthcare](https://www.udacity.com/course/big-data-analytics-in-healthcare--ud758)
  - [Spark](https://www.udacity.com/course/learn-spark-at-udacity--ud2002)
- **Databases and APIs**
  - [Data Wrangling with MongoDB](https://www.udacity.com/course/data-wrangling-with-mongodb--ud032)
  - [SQL for Data Analysis](https://www.udacity.com/course/sql-for-data-analysis--ud198)
  - [Designing RESTful APIs](https://www.udacity.com/course/designing-restful-apis--ud388)
- **Interview Preparation**
  - [Data Science Interview Prep](https://www.udacity.com/course/data-science-interview-prep--ud944)
  - [Machine Learning Interview Preparation](https://www.udacity.com/course/machine-learning-interview-prep--ud1001)


Mikel Sagardia, 2022.  
No guarantees.