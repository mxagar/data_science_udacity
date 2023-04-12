# Udacity Data Science Nanodegree: Data Engineering

These are my personal notes taken while following the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

The Nanodegree asssumes basic data analysis skills with python libraries (pandas, numpy, matplotlib, sklearn, etc.) and has 5 modules that build up on those skills:

1. Introduction to Data Science
2. Software Engineering
3. Data Engineering
4. Experimental Design & Recommendations
5. Data Scientist Capstone

This folder & guide refer to the **fifth module**: Capstone Project.

Mikel Sagardia, 2022.
No guarantees.

Overview of Contents:

- [Udacity Data Science Nanodegree: Data Engineering](#udacity-data-science-nanodegree-data-engineering)
  - [1. Project Overview](#1-project-overview)
  - [2. Projects](#2-projects)
  - [3. My Selected Project: Spark Project](#3-my-selected-project-spark-project)

## 1. Project Overview

Deliverables:

- Github repo with a proper `README.md`:
  - Description, Usage, Installation, Known Bugs, Author, License
- A blog post or a web app.

Steps:

- Define problem and possible solutions.
- EDA.
- Implement solution.
- Collect and visualize results.
- Write blog post or deploy web app.

Structure of the blog post (should be like a technical report):

- Section 1: Project definition
  - Overview: background, etc.
  - Problem statement: problem to be solved.
  - Metrics: explain why.
- Section 2: Analysis
  - Dataset: explain; account for distributions, outliers, etc.
  - Visualization.
- Section 3: Methodology
  - Data processing done
  - Implementation of the models, algorithms
  - Refinement: cross-validation, etc.
- Section 4: Results
  - Model evaluation and validation
  - Justification: choose best model/parameters, show charts
- Section 5: Conclusion
  - Reflection: summary and highlight particular aspects
  - Improvements

Interesting links:

- [How to Write a Good README File for Your GitHub Project](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
- [Writing a Scientific Report](https://www.waikato.ac.nz/library/guidance/guides/write-scientific-reports)
- [Components of a Scientific Report](https://canvas.hull.ac.uk/courses/370/pages/components-of-a-scientific-report)

## 2. Projects

List of suggested projects:

- Customer Segmentation Report for Arvato Financial Services: [Arvato Final Project](https://www.youtube.com/watch?v=qBR6A0IQXEE). **Note**: The data belongs to Bertelsmann/Arvato and must be deleted 2 weeks after the completion of the project.
- Optimizing App Offers With Starbucks: [Starbucks Capstone](https://www.youtube.com/watch?v=bq-H7M5BU3U).
- Use Convolutional Neural Networks to Identify Dog Breeds. I have already done this project: [project-dog-classification](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/project-dog-classification).
- Using Spark to Predict Churn with Insight Data Science: [Sparkify](https://www.youtube.com/watch?v=lPCzCEG2yRs). **Notes:**
  - A course needs to be done beforehand: [Spark](https://learn.udacity.com/courses/ud2002).
  - The provided workspace has a reduced version of the dataset of 128 MB; the original dataset has 12 GB.
  - The project should be deployed on a Spark cluster.

Apart from the suggested projects, we can also choose our own project.

Capstone project examples (Machine Learning Nanodegree):

- [Plot and Navigate a Virtual Maze](https://github.com/udacity/machine-learning/blob/master/projects/capstone/report-example-3.pdf)
- [Vision Loss](https://github.com/udacity/machine-learning/blob/master/projects/capstone/report-example-1.pdf)


## 3. My Selected Project: Spark Project

- Spark & Co. course notes [spark_big_data_guide](https://github.com/mxagar/spark_big_data_guide).
- Project repository: [sparkify_customer_churn](https://github.com/mxagar/sparkify_customer_churn).
