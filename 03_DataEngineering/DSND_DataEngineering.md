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
      - [Exercise 3: SQL](#exercise-3-sql)
      - [Exercise 4: APIs](#exercise-4-apis)
    - [3.3 Transform](#33-transform)
      - [Exercise 5: Combining Datasets](#exercise-5-combining-datasets)
      - [Exercise 6: Cleaning Data](#exercise-6-cleaning-data)
      - [Exercise 7: Data Types](#exercise-7-data-types)
      - [Exercise 8: Parsing Dates](#exercise-8-parsing-dates)
      - [Exercise 9: Encodings](#exercise-9-encodings)
      - [Exercise 10: Missing Values](#exercise-10-missing-values)
      - [Exercise 11: Duplicates](#exercise-11-duplicates)
      - [Exercise 12: Regex and Dummy Variables](#exercise-12-regex-and-dummy-variables)
      - [Exercises 13 and 14: Outliers](#exercises-13-and-14-outliers)
      - [Exercise 15: Scaling](#exercise-15-scaling)
      - [Exercise 16: Feature Engineering](#exercise-16-feature-engineering)
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
## JSON

import pandas as pd
# Orient:
# https://pandas.pydata.org/docs/reference/api/pandas.read_json.html
df_json = pd.read_json('population_data.json', orient='records')

## XML

# Often more manual processing is required
# Example XML file with these "record" objects and "fields" within:
# <record>
#   <field name="Country or Area" key="ABW">Aruba</field>
#   <field name="Item" key="SP.POP.TOTL">Population, total</field>
#   <field name="Year">1960</field>
#   <field name="Value">54211</field>
# </record>
# Parse with BeautifulSoup
from bs4 import BeautifulSoup
with open("population_data.xml") as fp:
    soup = BeautifulSoup(fp, "lxml") # lxml is the Parser type
# Convert the XML into dataframe
data_dictionary = {'Country or Area':[], 'Year':[], 'Item':[], 'Value':[]}
for record in soup.find_all('record'): # look for "record" objects
    for record in record.find_all('field'): # look for "field" objects
        data_dictionary[record['name']].append(record.text)
df = pd.DataFrame.from_dict(data_dictionary)
#   Country or Area	 Year	 Item	               Value
# 0	Aruba	           1960	 Population, total	 54211
# ...
# We need to / can pivot the table for better format
df = df.pivot(index='Country or Area', columns='Year', values='Value')
df.reset_index(level=0, inplace=True)
#  	Country or Area	  1960	    1961	    1962	    1963	1964	...	2017
# 0	Afghanistan	      8996351	  9166764	  9345868	  ...
# ...
```

#### Exercise 3: SQL

File: [`lab/03_sql/3_sql_exercise.ipynb`](lab/03_sql/3_sql_exercise.ipynb)

There are many ways of getting data from SQL databases. Here two are shown using a SQLite database:

- Using `sqlite3`, the python library for SQLite
- Using SQLAlchemy with raw SQL statements; there are other ways with SQLAlchemy, too.

SQLite creates databases in a single file (non-distributed); these storage is though for single applications, when we'd like to query data.

Content:

```python
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# SQLite
# Connect to the database
conn = sqlite3.connect('population_data.db')
# Run a query: there is only one table, population_data, and we extract everything
df = pd.read_sql('SELECT * FROM population_data', conn)

# SQLAlchemy
# Create an engine
engine = create_engine('sqlite:///population_data.db')
# Run SELECT * query
df = pd.read_sql("SELECT * FROM population_data", engine)

# Write to SQLite
conn = sqlite3.connect('dataset.db')
df = pd.read_csv('dataset.csv')
# Clean
columns = [col.replace(' ', '_') for col in df.columns]
df.columns = columns
df.to_sql("dataset", conn, if_exists="replace")
```

#### Exercise 4: APIs

World Bank APIs resources:

- [Documentation included how to filter by year](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structure)
- [2-character iso country codes](https://www.nationsonline.org/oneworld/country_code_list.htm)
- [Search box for World Bank indicators](https://data.worldbank.org)

To find the indicator code:

- First search for the indicator here: [https://data.worldbank.org](https://data.worldbank.org)
- Click on the indicator name. The indicator code is in the url.
- For example, the indicator code for total population is `SP.POP.TOTL`, which you can see in the link [https://data.worldbank.org/indicator/SP.RUR.TOTL](https://data.worldbank.org/indicator/SP.RUR.TOTL).

File: [`lab/04_api/4_api_exercise.ipynb`](lab/04_api/4_api_exercise.ipynb)

Content:

```python
# Define URL: Rural population in Switzerland between 1995-2001
url = 'http://api.worldbank.org/v2/country/ch/indicator/SP.RUR.TOTL/?date=1995:2001&format=json&per_page=1000'
# Send the request
r = requests.get(url)
# Convert to JSON: first element is metadata
r_json = r.json()
df = pd.DataFrame(r_json[1])
```

### 3.3 Transform

Typical transformation operations:

- Combine different datasets in different formats: join/merge, pivot/melt
- Cleaning data
- Checking data types
- Matching encodings
- Dealing with missing data
- Dealing with duplicates
- Encoding data: dummies
- Outliers
- Scaling data
- Feature engineering

#### Exercise 5: Combining Datasets

Typical pandas methods to combine datasets:

- Concatenate: `concat`
- Join: `merge`
- Pivot/unpivot between long/wide formats: `pivot`, `melt`

File: [`lab/05_combine_data/5_combining_data.ipynb`](lab/05_combine_data/5_combining_data.ipynb)

Content:

```python
# df_rural.columns = 'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', ..., '2017'
df_rural = pd.read_csv('rural_population_percent.csv', skiprows=4)
# df_electricity.columns = 'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', ..., '2017'
df_electricity = pd.read_csv('electricity_access_percent.csv', skiprows=4)

# New format: long
# df_rural.columns = 'Country Name, 'Country Code', 'Indicator Name', 'Indicator Code', 'Year', 'Rural Value'
df_rural = pd.melt(df_rural, id_vars=['Country Name',
                                      'Country Code',
                                      'Indicator Name',
                                      'Indicator Code'],
                             var_name='Year',
                             value_name='Rural Value')
# df_electricity.columns = 'Country Name, 'Country Code', 'Indicator Name', 'Indicator Code', 'Year', 'Electricity Value'
df_electricity = pd.melt(df_electricity, id_vars=['Country Name',
                                                  'Country Code',
                                                  'Indicator Name',
                                                  'Indicator Code'],
                                         var_name='Year',
                                         value_name='Electricity Value')

# Drop any columns from the data frames that aren't needed
df_rural.drop(['Indicator Name', 'Indicator Code'], axis=1, inplace=True)
df_electricity.drop(['Indicator Name', 'Indicator Code'], axis=1, inplace=True)

# Merge the data frames together based on their common columns
# in this case, the common columns are Country Name, Country Code, and Year
df_merge = df_rural.merge(df_electricity, how='outer',
                                          on=['Country Name', 'Country Code', 'Year'])

# Sort the results by country and then by year
df_combined = df_merge.sort_values(by=['Country Name', 'Year'])
df_combined.head()
```

#### Exercise 6: Cleaning Data

Typical data errors that need to be cleaned:

- data entry mistakes
- duplicate data
- incomplete records
- inconsistencies between dataset

File: [`lab/06_cleaningdata/6_cleaning_data.ipynb`](lab/06_cleaningdata/6_cleaning_data.ipynb).

It's an interesting but very specific case of data cleaning: country names are mapped to their ISO country codes:

- The library `pycountry` is used to get the ISO code given the official name.
- Countries that are not found in the `pycountry` database are mapped manually.
- Mapping is done with `.apply(lambda x: d[x])`, where `d` is a dictionary which maps `name` to `code`.

#### Exercise 7: Data Types

File: [`lab/07_datatypes/7_datatypes_exercis.ipynb`](lab/07_datatypes/7_datatypes_exercise.ipynb).

Contents:

- Column filtering is done with `isin()`.
- Filtered column values are summed with `sum(axis=0)`.
- String columns are converted to numeric by removing `,` first using `replace()` and then `to_numeric()`.

#### Exercise 8: Parsing Dates

Parsing dates is a common activity both in pandas and in [SQL](http://www-db.deis.unibo.it/courses/TW/DOCS/w3schools/sql/sql_dates.asp.html#gsc.tab=0).

File: [`lab/08_parsingdates/8_parsingdates_exercise.ipynb`](lab/08_parsingdates/8_parsingdates_exercise.ipynb).

Content:

```python
# Example closing date: 2023-06-28
# Format: https://strftime.org
df_projects['closingdate'] = pd.to_datetime(df_projects['closingdate'], format='%Y-%m-%dT%H:%M:%SZ')
# Get year, month, weekday
df_projects['closingdate'].dt.year
df_projects['closingdate'].dt.month
df_projects['closingdate'].dt.weekday
```

#### Exercise 9: Encodings

Text or file encodings are mappings between bytes and string symbols; the default encoding, which is also valid for all languages, is `utf-8`. But python also comes with other encodings, too: [Standard Encodings](https://docs.python.org/3/library/codecs.html#standard-encodings).

File: [`lab/09_encodings/9_encodings_exercise.ipynb`](lab/09_encodings/9_encodings_exercise.ipynb).

```python
from encodings.aliases import aliases

# When an encoding is not UFT-8, how to detect which encoding we should use?
# Python has a file containing a dictionary of encoding names and associated aliases
alias_values = set(aliases.values())
for alias in alias_values:
    try:
        df = pd.read_csv('mystery.csv', encoding=alias)
        print(alias) # valid encodings are printed
    except:
         pass 

# Another option: chardet
# !pip install chardet
import chardet

with open("mystery.csv", 'rb') as file:
    print(chardet.detect(file.read()))
```

#### Exercise 10: Missing Values

Most machine learning algorithms cannot handle missing values; exception: [Gradient Boosting: XGBoost](https://xgboost.readthedocs.io/en/latest/).

There are two ways to handle missing values:

- Delete data
  - If a column is almost all `NA`, we could consider deleting it
  - If a row is almost all `NA`, we could consider deleting it
- Fill missing values = **imputation**
  - **Mean / median / mode imputation** is an option if the missing cells are not that many; note that instead of imputing the column aggregate, we could group by other categorical features and impute the aggregate of that group!
  - Time series: **Forward fill, Backward fill**: if the data is ordered in time, we apply *hold last sample* in one direction or the other.

File: [`lab/10_imputation/10_imputations_exercise.ipynb`](lab/10_imputation/10_imputations_exercise.ipynb).

```python
# Imputation of regular features (not time series): fill with mean / median / mode
# WARNING: instead of imputing the column aggregate,
# we should group by other categorical features and impute the aggregate of that group!
df["var_fill"] = df.groupby("var_group")["var_fill"].transform(lambda x: x.fillna(x.mean())

# Imputation in time series: Forward Fill and Backward Fill
# i.e., if the data is ordered in time, we apply *hold last sample* 
# in one direction or the other. BUT: we need to sort the data!
df['GDP_ffill'] = df.sort_values(by='year').groupby("country")['GDP'].fillna(method='ffill')
df['GDP_bfill'] = df.sort_values(by='year').groupby("country")['GDP'].fillna(method='bfill')
# If only a country
df['GDP_ffill'] = df.sort_values(by='year')['GDP'].fillna(method='ffill')
# If the first/last value is NA, we need to run both: ffill and bfill
df['GDP_ff_bf'] = df.sort_values(by='year')['GDP'].fillna(method='ffill').fillna(method='bfill')

```

#### Exercise 11: Duplicates

This exercise goes beyond the typical `.duplicated()` and `.drop_duplicates().reset_index(drop=True)`. It analyzes the case of projects in Yogoslavia and the countries in which Yogoslavia was segregated. The idea is that some projects might be (and, in fact, are) duplicated. These are found by checking the country names and the project approval dates.

File: [`lab/11_duplicatedata/11_duplicatedata_exercise.ipynb`](lab/11_duplicatedata/11_duplicatedata_exercise.ipynb).

#### Exercise 12: Regex and Dummy Variables

Categorical variables need to be converted into numbers; one approach is using dummy variables. However, categories often need to be cleaned and reduced (i.e., aggregated), otherwise the number of dummy columns explode. A common way of cleaning consists in using `.replace()` together with `re`, the regex module. More information on regex:

- [Regex tutorial — A quick cheatsheet by examples](https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285)
- [Regex cookbook — Top 15 Most common regex](https://medium.com/@fox.jonny/regex-cookbook-most-wanted-regex-aa721558c3c1)

File: [`lab/12_dummy_variables/12_dummyvariables_exercise.ipynb`](lab/12_dummy_variables/12_dummyvariables_exercise.ipynb).

```python
## Cleaning categories

# Fields with value '!$10' -> NaN
df['sector'] = df['sector'].replace('!$10', np.nan)
# Replace with Regex
# This looks for string with an exclamation point followed by one or more characters
df['sector'] = df['sector'].replace('!.+', '', regex=True)
# Replace with Regex
# Remove the string '(Historic)' from the sector1 variable
df['sector'] = df['sector'].replace('^(\(Historic\))', '', regex=True)
# More on regex:
# - Tutorial: https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285
# - Cookbook: https://medium.com/@fox.jonny/regex-cookbook-most-wanted-regex-aa721558c3c1

## Aggregating categories

import re

# Create an aggregate sector variable which covers general topics
# For instance: "Transportation: Highways", "Transportation: Airports" -> "Transportation"
df.loc[:,'sector_aggregates'] = sector['sector']
topics = ['Energy', 'Transportation']
for topic in topics:
    # Find all that contain the topic (ignore case), replace NaN with False (i.e., not found)
    # All found have same general topic
    df.loc[sector['sector_aggregates'].str.contains(topic, re.IGNORECASE).replace(np.nan, False),'sector_aggregates'] = topic

## Dummy Variables

# One-hot encoding of features: Dummy variables with pandas
# Use drop_first=True to remove the first category and avoid multi-colinearity
# Note: if a field has NaN, all dummy variables will have a value of 0
col_dummies = ['var1', 'var2']
try:
    for col in col_dummies:
        df = pd.concat([df.drop(col, axis=1),
        				pd.get_dummies(df[col], prefix=col, prefix_sep='_',
        					drop_first=True, dummy_na=False)],
                        axis=1)
except KeyError as err:
    print("Columns already dummified!")

```

#### Exercises 13 and 14: Outliers

Outliers are data points that have unexpected values; they can be due to:

- Errors in the recordings, i.e., we should remove them
- Due to chance, i.e., we should keep them

There are many ways to detect them:

- Data visualization: in 1D or 2D, plot as visually inspect; in higher dimensions, apply PCA to 2D and inspect.
- Clustering (in any dimension): cluster the data and compute distances to centroids; values with large distances are suspicious of being outliers.
- Statistical methods:
  - Z-score (assuming normal distribution): any data point outside from the the 2-sigma range is an outlier (i.e., `< mean-2*sigma` or `> mean+2*sigma`); 2-sigma is related to the 95% Ci or `alpha = 0.05`. 
  - Tukey method (no distribution assumption): any data point outside from the 1.5*IQR is an outlier (i.e., `< Q1-1.5*IQR` or `> Q3+1.5*IQR`)

Outliers might affect the model considerably; for instance, in a linear regression model, the line/hyperplane is pulled to the outliers. However, sometimes the outliers are aligned with the model, and they are not really outliers; when should we remove them?

- Compute candidate outliers in all dimensions, e.g., using Tukey.
- Candidate data points that are outliers in all dimensions are maybe not outliers.
- Create models with and without candidate outliers and then predict control points; do they change considerably?

Outlier detection in Scikit-Learn: [Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html).

Files: 

- [`lab/13_outliers_part1/13_outliers_exercise.ipynb`](lab/13_outliers_part1/13_outliers_exercise.ipynb).
- [`lab/14_outliers_part2/14_outliers_exercise.ipynb`](lab/14_outliers_part2/14_outliers_exercise.ipynb).

```python
def tukey_filter(df, col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    max_value = Q3 + 1.5 * IQR
    min_value = Q1 - 1.5 * IQR
    return df[(df[col_name] < max_value) & (df[col_name] > min_value)]
```

#### Exercise 15: Scaling

Algorithms that use Euclidean distance computations work with data in similar ranges; thus, scaling is necessary; typical scaling methods:

- Rescaling: scale values to `[0,1]`, aka. `MinMaxScaling`.
- Standardization: scale to get a mean of 0 and a standard deviation of 1, aka. `StandardScaling`.

File: [`lab/15_scaling/15_scaling_exercise.ipynb`](lab/15_scaling/15_scaling_exercise.ipynb).

#### Exercise 16: Feature Engineering

Typical feature engineering approaches:

- Encode/map values
- Ratios
- Sums
- Polynomial features: `PolynomialFeatures`
- Transform data into a new feature

File: [`lab/16_featureengineering/16_featureengineering_exercise.ipynb`](lab/16_featureengineering/16_featureengineering_exercise.ipynb).

### 3.4 Load

The last step in an ETL pipeline is loading the dataset we have prepared.



## 3. NLP Pipelines

## 4. Machine Learning Pipelines

## 5. Project: Disaster Response Pipeline
