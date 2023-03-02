# Udacity Data Science Nanodegree: Data Engineering

These are my personal notes taken while following the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

The Nanodegree asssumes basic data analysis skills with python libraries (pandas, numpy, matplotlib, sklearn, etc.) and has 5 modules that build up on those skills:

1. Introduction to Data Science
2. Software Engineering
3. Data Engineering
4. Experimental Design & Recommendations
5. Data Scientist Capstone

This folder & guide refer to the **third module**: Data Engineering.

Mikel Sagardia, 2022.
No guarantees.

Overview of Contents:

- [Udacity Data Science Nanodegree: Data Engineering](#udacity-data-science-nanodegree-data-engineering)
  - [1. Introduction to Data Engineering](#1-introduction-to-data-engineering)
    - [Project Overview](#project-overview)
    - [Exercises and Code](#exercises-and-code)
  - [2. ETL Pipelines](#2-etl-pipelines)
    - [3.1 Lesson Outline and Dataset](#31-lesson-outline-and-dataset)
      - [Dataset: World Bank Data](#dataset-world-bank-data)
    - [3.2 Extract](#32-extract)
      - [Exercise 1: CSV](#exercise-1-csv)
      - [Exercise 2: JSON and XML](#exercise-2-json-and-xml)
    - [3.3 Transform](#33-transform)
    - [3.4 Load](#34-load)
  - [3. NLP Pipelines](#3-nlp-pipelines)
  - [4. Machine Learning Pipelines](#4-machine-learning-pipelines)
  - [5. Project: Disaster Response Pipeline](#5-project-disaster-response-pipeline)

## 1. Introduction to Data Engineering

Data engineers gather data from different sources, clean and process it, and store it for later use. Then, anyone can use those data without much wrangling. That processing done by data engineers can be automated with data pipelines.

Roles of a data engineer:

- In large companies there are dedicated data engineers, which are closer to a software engineer role.
- In smaller companies, the role is more diffuse and distributed among data scientists and software developers.

The module focuses on both cases!

### Project Overview

We have messages sent during disasters via social media or other means; we need to create:

- An ETL pipeline which processes messages and stores them to an SQLite database.
- A Machine Learning pipeline which contains a model that classifies the message in 36 categories.

Finally, a web app is created in Flask so that we can classify a new message using the ML pipeline.

This kind of systems are crucial, since at disaster events millions and millions of messages are generated, and there is no human capacity to process them all; thus, an automated system is needed.

### Exercises and Code

All exercises are in [`lab/`](./lab/).

## 2. ETL Pipelines

**ETL = Extract, Transform, Load**:

- Extract: get data from different sources
- Transform: clean and format data to fit a predefined schema
- Load: save the data prepared to be used

In a large company, a data engineer is responsible for the ETL pipeline; then, a data scientist takes the data saved by the data engineer to create the machine learning pipeline. However, in small companies, the data scientist might be responsible for the ETL pipeline as well.

Example: Log data: we use log data to create a database with clicks and regions; we need to parse the logs to get the timestamps and IPs.

Cloud computing has popularized another pipeline: **ELT = Extract, Load, Transform**; these are the so called **data warehouses**. A regular ETL pipeline prepares the data in a predefined format so that we can query the dataset/database easily many times; thus, the transformation happens once. In contrast, ELT pipelines allow to store data without transformations; instead, transformations occur when querying. We can still use SQL or SQL-like languages for the queries.

Note that small datasets are used in the lesson, but usually big datasets appear in the industry, so big that they don't fit on one server, i.e., they are distributed across different locations &mdash; that's big data. **Free Udacity courses** on **Big Data** topics:

- [Intro to Hadoop and MapReduce](https://www.udacity.com/course/intro-to-hadoop-and-mapreduce--ud617)
- [Deploying a Hadoop Cluster](https://www.udacity.com/course/deploying-a-hadoop-cluster--ud1000)
- [Real-Time Analytics with Apache Storm](https://www.udacity.com/course/real-time-analytics-with-apache-storm--ud381)
- [Big Data Analytics in Healthcare](https://www.udacity.com/course/big-data-analytics-in-healthcare--ud758)
- [Spark](https://www.udacity.com/course/learn-spark-at-udacity--ud2002)

Interesting links:

- [ETL Wikipedia](https://en.wikipedia.org/wiki/Extract,_transform,_load)
- Data warehouses:
  - [Amazon Redshift](https://aws.amazon.com/redshift/)
  - [Google BigQuery](https://cloud.google.com/bigquery/)
  - [IBM Db2 Warehouse on Cloud](https://www.ibm.com/cloud/db2-warehouse-on-cloud)

### 3.1 Lesson Outline and Dataset

In this lesson, the following topics are learned:

1. Extract data from different sources such as:
   
   - csv files
   - json files
   - APIs

2. Transform data

   - combining data from different sources
   - data cleaning
   - data types
   - parsing dates
   - file encodings
   - missing data
   - duplicate data
   - dummy variables
   - remove outliers
   - scaling features
   - engineering features

3. Load: send the transformed data to a database
4. ETL Pipeline: code an ETL pipeline

#### Dataset: World Bank Data

Data from two sources is used:

- [World Bank Indicator Data](https://data.worldbank.org/indicator): GDP, population, etc.
- [World Bank Projects & Operations Data](https://datacatalog.worldbank.org/search/dataset/0037800): money spent to build a bridge in Nepal, etc.

The data can be downloaded as CSV or access via an API.

The goal is to aggregate and clean all data, and bring them together in one table.

### 3.2 Extract

Extraction: pulling and gathering data from different sources, for instance:

- CSV
- JSON
- XML
- Text files, e.g., log files (NLP is required)
- SQL databases
- Web scrapping
- APIs: JSON/XML strings obtained

#### Exercise 1: CSV

File: [`lab/01_csv/1_csv_exercise_solution.ipynb`](lab/01_csv/1_csv_exercise_solution.ipynb)

Contents:

```python
# If messy data with various types in a col, convert to string
df_projects = pd.read_csv('projects_data.csv', dtype=str)
# Nulls for each column
df_projects.isnull().sum()
# If first rows have info lines, not CSV, skip them
df_population = pd.read_csv("population_data.csv", skiprows=4)
# Nulls for each row
df_population.isnull().sum(axis=1)
# Drop columns with many NAs or dubious content
df_population = df_population.drop(columns='Unnamed: 62')
```

#### Exercise 2: JSON and XML

JSON and XML are common exchange formats in APIs.

File: [`lab/02_json_xml/2_extract_exercise.ipynb`](lab/02_json_xml/2_extract_exercise.ipynb)

Contents:

```python

```

### 3.3 Transform

### 3.4 Load

## 3. NLP Pipelines

## 4. Machine Learning Pipelines

## 5. Project: Disaster Response Pipeline
