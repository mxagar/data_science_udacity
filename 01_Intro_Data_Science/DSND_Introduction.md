# Udacity Data Science Nanodegree: Introduction

These are my personal notes taken while following the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

Mikel Sagardia, 2022.
No guarantees.

Overview of Contents:

1. Welcome
  - 1.1 The Skills that Set You Apart
2. Introduction to Data Science
  - Lesson 2: The Data Science Process
    - Business Understanding & Data Understanding (CRISP-DM 1 & 2 / 6)
    - Preparing the Data - Missing Data: Removing (CRISP-DM 3 / 6)
    - Preparing the Data - Missing Data: Imputing (CRISP-DM 3 / 6)
    - Preparing the Data - Working with Categorical Variables (CRISP-DM 3 / 6)
    - Modelling - Predicting the Salary (CRISP-DM 4 / 6)
    - Modelling - Overfitting (CRISP-DM 4 / 6)
    - Results (CRISP-DM 5 / 6)
    - Deploy (CRISP-DM 6 / 6)
    - Notebooks
      1. `A Look at the Data.ipynb`
        - Dataset is loaded, shape and variables analyzed
        - Light EDA on the education of the participants
      2. `How To Break Into the Field.ipynb`
        - Values of `CousinEducation` are cleaned
        - Stratification of participants by `Education` to see differences in `CousinEducation`
      3. `Job Satisfaction.ipynb`
        - `JobSatisfaction` is analyzed by stratifying/grouping according to `EmploymentStatus`, `CompanySize`, `HomeRemote`, `FormalEducation`
      4. `What Happened.ipynb`
        - Correlation heatmap computed: numeric variables, target
      5. `Removing Values.ipynb`
        - Different examples of `drop()` and `dropna()`
      6. `Removing Data Part II.ipynb`
        - Rows with NA target value must be removed
        - Different strategies followed to remove rows/data-points or columns/features
      7. `Imputation Methods and Resources -.ipynb`
        - Different examples of `fillna()` and `apply()`
      8. `Imputing Values.ipynb`
        - Mean is imputed for NA values
        - The first model with only numerical variables is created; it has a low R2
      9. `Categorical Variables.ipynb`
        - Categorical variables are detected
        - Categoricals are encoded as dummy variables
        - A new model is built and fit; better performance, but it's overfitting
      10. `Putting It All Together.ipynb`
        - Everything is functionalized: data cleaning (removing, imputation)
        - Models are trained with decreasing number of features: less feeatures, less overfitting
        - The optimum model is chosen
        - The salary prediction coefficients are shown, sorted
    - Notebooks: Summary of Interesting Python Commands
    - Concepts to Review, Take-Aways, Caveats
  - Lesson 3: Communicating to Stakeholders


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
  - Exploratory Data Analysis
3. Data preparation **(80% of the time goes here!)**
  - Data cleaning: missing values
  - Feature engineering: transformations, categorical encoding, scaling
  - Outliers
4. Data modeling
  - Select model
  - Fit and evaluate
  - Tune parameters, check required features for less overfitting
5. Result evaluation
6. Deployment
  - Deploy for use
  - Communicate results to stakeholders

This lesson uses an example dataset:

- my local folder:
  - original: `~/git_repositories/cd0017-introduction-to-data-science`
  - link: `./lab/CRISP_DM`
- forked repository: [https://github.com/mxagar/cd0017-introduction-to-data-science](https://github.com/mxagar/cd0017-introduction-to-data-science)
- the dataset: `./lab/data`; however, the forked repo ha sits own version, which seems to be the correct one for the test answers

The dataset seems to be a survey from Stackoverflow to their users. 154 variables have been collected for 19102 individuals. Two files are provided for the dataset:

- `survey_results_schema.csv`: the 154 variables and their definition
- `survey_results_public.csv`: 19102 x 154 dataset

### Business Understanding & Data Understanding (CRISP-DM 1 & 2 / 6)

We need to select the business questions we're interested in and selectthe variables that would help answer them.

For instance:

- How can we enter the field? `CousinEducation`
- What are job placement and salary rates for bootcamps? `TimeAfterBootcamp`
- What relates to salary? Any or all columns
- What relates to job satisfaction? Any or all columns

### Preparing the Data - Missing Data: Removing (CRISP-DM 3 / 6)

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

### Preparing the Data - Missing Data: Imputing (CRISP-DM 3 / 6)

Another way of dealing with missing data is to impute or assign values: 

- Typical imputations: mean (quantitative), median (quantitave, skewed), mode (categorical)
- More advanced imputations:
  - Regression of value considering other features which are available
  - Observe similar data points and pick their feature value; k nearest neighbors (kNN) is used

However, very important: **when imputing a value, we are diluting its importance!** We are artificially making the rows more similar, when that might not be the case. Thus, always be very careful and think of the effects of what we're doing.

### Preparing the Data - Working with Categorical Variables (CRISP-DM 3 / 6)

Using one-hot encoding or dummy variables is easy and has the advantage that we can easily interpret the influence of a category in the output when using linear models. However, if we have many categorical variables with many levels/categories, the number of dummy variables explodes.

Additionally, each dummy variable should have at least 10 data points with a `1` value in it; that's a rule of thumb that is not always met.

### Modelling - Predicting the Salary (CRISP-DM 4 / 6)

Modelling implies creating supervised learning models that predict either by regression or classification.
However, note that inferences do not need models.

### Modelling - Overfitting (CRISP-DM 4 / 6)

When the model fails to generalize with the test split, we have overfitting: the accuracy or the used metric is considerably lower in the test split.

Methods to avoid overfitting:

- Regularization
- Reduce the number of features: try subsets of features
- Fit the model many times with different rows, then average the responses; aka. bootstraping

If we have overfittting, do not add more features! Instead, reduce them!

### Results (CRISP-DM 5 / 6)

We just answer the business questions.

### Deploy (CRISP-DM 6 / 6)

We either deploy to production (seen in later modules) or communicate the findings to the stakeholders (next lesson).

For communication:

- Github
- Medium, Blog
- Dashboard

### Notebooks

The notebooks should depict the usual processs in data science, although I have seen better introductory courses & notebooks than these. The dataset is the one introduced above: Stackoverflow survey on salaries and job satisfaction. In the following, I summarize their content in the order they should be done. Below, a collection of helpful code snippets extracted from the notebooks is provided.

The repository: [https://github.com/mxagar/cd0017-introduction-to-data-science](https://github.com/mxagar/cd0017-introduction-to-data-science)

1. `A Look at the Data.ipynb`
  - Dataset is loaded, shape and variables analyzed
  - Light EDA on the education of the participants
2. `How To Break Into the Field.ipynb`
  - Values of `CousinEducation` are cleaned
  - Stratification of participants by `Education` to see differences in `CousinEducation`
3. `Job Satisfaction.ipynb`
  - `JobSatisfaction` is analyzed by stratifying/grouping according to `EmploymentStatus`, `CompanySize`, `HomeRemote`, `FormalEducation`
4. `What Happened.ipynb`
  - Correlation heatmap computed: numeric variables, target
5. `Removing Values.ipynb`
  - Different examples of `drop()` and `dropna()`
6. `Removing Data Part II.ipynb`
  - Rows with NA target value must be removed
  - Different strategies followed to remove rows/data-points or columns/features
7. `Imputation Methods and Resources -.ipynb`
  - Different examples of `fillna()` and `apply()`
8. `Imputing Values.ipynb`
  - Mean is imputed for NA values
  - The first model with only numerical variables is created; it has a low R2
9. `Categorical Variables.ipynb`
  - Categorical variables are detected
  - Categoricals are encoded as dummy variables
  - A new model is built and fit; better performance, but it's overfitting
10. `Putting It All Together.ipynb`
  - Everything is functionalized: data cleaning (removing, imputation)
  - Models are trained with decreasing number of features: less feeatures, less overfitting
  - The optimum model is chosen
  - The salary prediction coefficients are shown, sorted

### Notebooks: Summary of Interesting Python Commands

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

# Get coefficient values from linear model
coefs_df = pd.DataFrame()
coefs_df['estimators'] = X_train.columns
coefs_df['coefs'] = lm.coef_
coefs_df['abs_coefs'] = np.abs(lm.coef_)
coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)

# Imputation
fill_mean = lambda col: col.fillna(col.mean())
df = df.apply(fill_mean, axis=0) # apply to axis = 0, ie., rows; NOT inplace
df['A'].fillna(df['A'].mean()) # another easier way
fill_mode = lambda col: col.fillna(col.mode()[0]) # mode() returns a series, pick first value
df = df.apply(fill_mode, axis=0)
# IMPORTANT NOTE: Consider better this approach
# Because apply might lead to errors
num_vars = df.select_dtypes(include=['float', 'int']).columns
for col in num_vars:
    df[col].fillna((df[col].mean()), inplace=True)

# One-hot encoding / Dummy variables
# Select categorical variables
cat_df = df.select_dtypes(include=['object']).copy()
# The number of columns with no missing values
len(cat_df.columns[cat_df.notna().sum() == cat_df.shape[0]])
# The number of columns with more than half of the column missing
len(cat_df.columns[cat_df.isna().sum()/cat_df.shape[0] > 0.5])
# Create a dummy variable NaN for missing values: Maybe it's informative
# and add prefix _ to column names
# IMPORTANT NOTE: apparently it is better to do it column by column
cat_df_dummy = pd.get_dummies(cat_df['col'], prefix_sep='_', dummy_na=True)
# Concatenate numeric/non-categorical & dummies
df_new = pd.concat([cat_df.drop('col',axis=1),cat_df_dummy],axis=1)
# Column by column; note: efficiency could be improved!
cat_df = df.select_dtypes(include=['object'])
cat_cols = cat_df.columns
for col in cat_cols:
  df = pd.concat([df.drop(col, axis=1), 
                  pd.get_dummies(
                                df[col],
                                prefix=col, 
                                prefix_sep='_',
                                drop_first=True,
                                dummy_na=False)],
                  axis=1)
```

### Concepts to Review, Take-Aways, Caveats

- Confidence intervals: review
- It's not always about ML; sometimes curiosity, proper data and analysis are enough to answer the business questions
- If a correlation matrix has a missing cell, it means that if the one column had a value, the other didn't! That means the model will fail if we feed those two columns without dealing with the missing values.
- Review: Data pre-processing notebooks by Soledad Galli. These contain a very nice summary of what is to be done in the pre-processing steps!
- When we drop NAs, we drop entire rows; check first the percentages of missing values in columns: maybe it makes more sense to ddrop columns instead of rows!
- Use `try-execpt` in parts of the code where it could fail, e.g., when fitting models or scoring.
- `pandas.isnull()` is an alias of `pandas.isna()`, so they do exactly the same thing!

## Lesson 3: Communicating to Stakeholders

This lesson is about

- how to publish on Github
- how to publish on Medium

### README Files

Anatomy of a `README.md` (not all parts are necessary):

- Title
  - Short description
- Installation / Getting Started
  - Dependencies
  - Installation commands
- Usage
  - Commands
  - Known bugs
- Contributing
  - Guidelines if people wantto contribute
- Code Status
  - are all tests passing?
  - shields: build/passing
  - if necessary
- FAQs (if necessary)
- License / Copyright
  - By default, I have the intelectual property, but it's not bad stating it explicitly if necessary
  - Choose appropriate license

Examples of READMEs:

- [Bootstrap](https://github.com/twbs/bootstrap)
- [Scikit Learn](https://github.com/scikit-learn/scikit-learn)
- [Stackoverflow blog](https://github.com/jjrunner/stackoverflow)

Observations:

- Files should appear as links: `[File](file.md)`
- README sections should appear as anchor links: `[My Heading](#my-heading)`
- If we have a section/part which is long and could be collapsed, do it! Use `<details>` and `<summary>`; see below.
- Provide all links possible in text
- Big projects have
  - Documentation links
  - Shields as Status
  - How to contribute
  - Community
  - Sponsors
  - Authors / Creators
  - Thanks
  - Backers

Example of **callapsable text**:

```Markdown
For some amazing info, you can checkout the [below](#amazing-info) section.

<details><summary>Click to expand</summary>

## Amazing Info
It would be great if a hyperlink could directly show this title.

</details>
```

### Github Basics

```bash
git clone
git add *
git commit -m "message"
git push
```

### Medium

Interesting links:

- [How to Use Medium: A Beginner's Guide to Writing, Publishing & Promoting on the Platform](https://blog.hubspot.com/marketing/how-to-use-medium)
- [Tips and Tricks for Medium Writers](https://blog.medium.com/tips-and-tricks-for-medium-writers-1d79498101c3)
- [How to attract users with headlines](https://medium.com/the-mission/this-new-data-will-make-you-rethink-how-you-write-headlines-751358f6639a)
- [A Scientific Guide to Posting Tweets, Facebook Posts, Emails, and Blog Posts at the Best Time](https://buffer.com/resources/best-time-to-tweet-post-to-facebook-send-emails-publish-blogposts/)
- [The Art of Storytelling in Analytics and Data Science](https://www.analyticsvidhya.com/blog/2020/05/art-storytelling-analytics-data-science/)
- [Making Money on Medium](https://medium.com/words-for-life/a-100-transparent-look-at-my-first-medium-paycheck-197b69483b44)
- [Can You Make Money on Medium?](https://writingcooperative.com/can-you-make-money-on-medium-90565989d599)


### Know Your Audience

The audience shoudl relate to the content.

Technical audience:

- Technical details
- Edge cases, difficulties
- Creative solutions

Broad audience:

- Questions of interest
- Results of each question
- Interesting findings through visuals

### Three Steps to Captivate Your Audience

1. Pull in and engage
2. Keep engaged with: strong storytelling, personal voice, article structure
3. Provide a practical ending with the desired takeaways + Call to Action

#### 1. Pull in and engage

On Medium, the readers see **title** + **image**.

We must have a compelling image to grab our reader! The title supports it. Spend time on the image! Think of a captivating image and then fnd a title to match; in that order!

#### 2. Keep engaged

Once the reader is on the page, we have 2 sentences and one image to keep him/her. A personal voice is a useful tactic for that; some possible methods:

- Tell a personal story
- Ask a question we had, and that we think the audience might have, too
- Relate to current events that are shared by the audience

Break up the content to 3 lines / idea, never more than 5 lineas / idea. Avoid long blocks without visuals or white spaces.

#### 3. Provide a practical ending


