# Udacity Data Science Nanodegree: Introduction

These are my personal notes taken while following the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

Mikel Sagardia, 2022.
No guarantees.

Overview of Contents:

1. Welcome

# 1. Welcome

## 1.1 The Skills that Set You Apart

Robert Chang, AirBnB

- Focus on fundamentals: Statistics, dealing with computers, etc.
- AirBnB tracks
  - Analytics: work with decision takers, communication
  - Inference: experiment design, A/B tests, tracking metrics, product analytics
  - Algorithm: build data products: recommendation engines, etc.

Dan Frank, Coinbase

- Typical Data Science roles:
  - Algorithms, Machine Learning: C++, Scikit-Learn, Tensorflow
  - Experimentation, more generalist: data engineering, Spark, SQL
  - High level business analysis: dashboarding tools, visualization

Richard Sharp, Starbucks

- Needed: Maths, Progamming (not always need formal training)

# 2. Introduction to Data Science

## Lesson 2: The Data Science Process

The CRISP-DM Process = Cross-Industry Standard Process for Data Mining: Standardized steps used in the data science process; always try to look back and detect in which step of CRSIP-DM we are!

1. Business understanding
  - Which are the questions we want to answer?
  - How do these questions make sense?
2. Data understanding
  - Which data do we need to collect to answer the question?
3. Data preparation
4. Data modeling
5. Result evaluation
6. Deployment

This lesson uses an example dataset:

- my local folder:
  - original: `~/git_repositories/cd0017-introduction-to-data-science`
  - link: `./lab/CRISP_DM`
- forked repository: [https://github.com/mxagar/cd0017-introduction-to-data-science](https://github.com/mxagar/cd0017-introduction-to-data-science)
- the dataset: `./lab/data`; however, the forked repo ha sits own version, which seems to be the correct one for the test answers

The dataset seems to be a survey from Stackoverflow to their users. 154 variables have been collected for 19102 individuals. Two files are provided for the dataset:

- `survey_results_schema.csv`: the 154 variables and their definition
- `survey_results_public.csv`: 19102 x 154 dataset

### Business Understanding & Data Understanding

We need to select the business questions we're interested in and selectthe variables that would help answer them.

For instance:

- How can we enter the field? `CousinEducation`
- What are job placement and salary rates for bootcamps? `TimeAfterBootcamp`
- What relates to salary? Any or all columns
- What relates to job satisfaction? Any or all columns

### Modelling: Predicting the Salary

Modelling implies creating supervised learning models that predict either by regression or classification.
However, note that inferences do not need models.

### Missing Data: Removing

Models usually cannot deal with missing data; therefore, we need to process NA values. The three major ways:

- Drop rows wwith any missing value
- Impute missing values
- Work around

We should always ask ourselves why there are missing data. Just droping rows or imputing values might lead to biased/unbalanced datasets! Example: lacking answers to a dating survey might indicate a certain type of personality. Thus, is is useful to keep track of the missing values:
  
- Create a new column with `missing_var_x =  1 | 0`
- Create a new column which counts the total number of missing answers by a person

When should we drop data with missing values? Think on the dtaa collection process. Examples:

- We collect the speed of a person running/walking with GPS; in a forest, there's no missing data. Better remove those data points than impute them
- We collect expected car prices from people: whean person does not answer, it might mean he/she doesn't own a car. That is information! Do not simply drop, track it as in the example above.

Taking decisions based on the number of missing values:

- If a small percentage of values is missing (< 5%) we could consider imputing them
- If a large percentage of values is missing (> 70%) we should consider removing the column, we see if the provided data is useful
- If the percentage is considerable (50%) consider removing it, but create a dummy column which tracks whether missing column or not

### Missing Data: Imputing

Another way of dealing with missing data is to impute or assign values: 

- Typical imputations: mean (quantitative), median (quantitave, skewed), mode (categorical)
- More advanced imputations:
  - Regression of value considering other features which are available
  - Observe similar data points and pick their feature value; k nearest neighbors (kNN) is used

However, very important: **when imputing a value, we are diluting its importance!** We are artificially making the rows more similar, when that might not be the case. Thus, always be very careful and think of the effects of what we're doing.

### Concepts to Review and Take Aways

- Confidence intervals: review
- It's not always about ML; sometimes curiosity, proper data and analysis are enough to answer the business questions
- If a correlation matrix has a missing cell, it means that if the one column had a value, the other didn't! That means the model will fail if we feed those two columns without dealing with the missing values.
- Review: Data pre-processing notebooks by Soledad Galli. These contain a very nice summary of what is to be done in the pre-processing steps!
- When we drop NAs, we drop entire rows; check first the percentages of missing values in columns: maybe it makes more sense to ddrop columns instead of rows!
- Use `try-execpt` in parts of the code where it could fail, e.g., when fitting models or scoring.
- `pandas.isnull()` is an alias of `pandas.isna()`, so they do exactly the same thing!

### Notebooks

The notebooks should depict the usual processs in data science, although I have seen better introductory courses & notebooks than these. The dataset is the one introduced above: Stackoverflow survey on salariey and job satisfaction.

1. `A Look at the Data.ipynb`
  - a
  - b
2. `How To Break Into the Field.ipynb`
  - a
  - b
3. `Job Satisfaction.ipynb`
  - a
  - b
4. `What Happened.ipynb`
  - a
  - b
5. `Removing Values.ipynb`
  - a
  - b
6. `Removing Data Part II.ipynb`
  - a
  - Rows with NA target value must be removed
7. `Imputation Methods and Resources -.ipynb`
  - a
  - b

#### Summary of Interesting Python Commands

This section collects the most interesting python commands from the notebooks:

```python
# Columns/Feature with NO missing values
no_nulls = set(df.columns[df.isnull().sum()==0])

# Percentage of people that reported salary
df['Salary'].notna().sum() / df.shape[0]

# Columns/Feature with more than 75% of values missing
most_missing_cols = set(df.columns[(df.isnull().sum()/df.shape[0]) > 0.75])

# Level frequency distributions of categorical values: Bar charts
status_vals = df['Professional'].value_counts()
(status_vals/df.shape[0]).plot(kind="bar");

# Compare value_counts frequencies of two subgroups
# using pandas styling
# https://pandas.pydata.org/pandas-docs/stable/style.html
df['Group'] = df["Variable"].apply(...) # create value 0, 1 depending on "Variable"
df_0 = df[df['Group']==0] # subset 0
df_1 = df[df['Group']==1] # subset 1
# Merge dataframes of value_counts tables
comp_df = pd.merge( df["Variable"].value_counts().reset_index(),
                    df["Variable"].value_counts().reset_index(),
                    left_index=True, right_index=True)
comp_df.columns = ['1', '0']
comp_df['Diff'] = comp_df['1'] - comp_df['0']
# Pandas styling
comp_df.style.bar(subset=['Diff'], align='mid', color=['#d65f5f', '#5fba7d'])

# Average job satisfaction depending on company size
df.groupby('CompanySize')['JobSatisfaction'].mean().dropna().sort_values()

# Values, histograms and heatmap of the correlation of quantitative variables
df.describe()
df.hist()
sns.heatmap(df.corr(), annot=True, fmt=".2f")

# Dropping missing values
all_drop  = small_dataset.dropna() # all rows with at least one NA columns dropped
all_row = small_dataset.dropna(how='all') # all rows with all cols NA dropped
only3or1_drop = small_dataset.dropna(subset=['col1','col3']) # all rows with at least one NA in col1 or col3 dropped
df.drop('C',axis=1,inplace=True) # drop complete column C in place; default axis = 0, ie., rows

# Linear Regression snippet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state=42)
lm = LinearRegression(normalize=True) # better, use sklearn StandardScaler
lm.fit(X_train, y_train)
y_test_preds = lm_model.predict(X_test)
r2_test =  r2_score(y_test,y_test_preds)

# Imputation
fill_mean = lambda col: col.fillna(col.mean())
df.apply(fill_mean, axis=0) # apply to axis = 0, ie., rows; NOT inplace
df['A'].fillna(df['A'].mean()) # another easier way

```
