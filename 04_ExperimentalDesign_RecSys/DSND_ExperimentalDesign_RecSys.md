# Udacity Data Science Nanodegree: Data Engineering

These are my personal notes taken while following the [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

The Nanodegree assumes basic data analysis skills with python libraries (pandas, numpy, matplotlib, sklearn, etc.) and has 5 modules that build up on those skills:

1. Introduction to Data Science
2. Software Engineering
3. Data Engineering
4. Experimental Design & Recommendations
5. Data Scientist Capstone

This folder & guide refer to the **fourth module**: Experimental Design & Recommendations.

Mikel Sagardia, 2022.
No guarantees.

Overview of Contents:

- [Udacity Data Science Nanodegree: Data Engineering](#udacity-data-science-nanodegree-data-engineering)
  - [1. Introduction to Experiment Design and Recommendation Engines](#1-introduction-to-experiment-design-and-recommendation-engines)
  - [2. Concepts in Experiment Design](#2-concepts-in-experiment-design)
    - [2.1 What Is an Experiment?](#21-what-is-an-experiment)
    - [2.2 Types of Experiment](#22-types-of-experiment)
    - [2.3 SMART Experiments](#23-smart-experiments)
    - [2.4 Types of Sampling](#24-types-of-sampling)
    - [2.5 Measuring Outcomes](#25-measuring-outcomes)
    - [2.6 Creating Metrics](#26-creating-metrics)
    - [2.7 Controlling Variables](#27-controlling-variables)
    - [2.8 Checking Validity](#28-checking-validity)
    - [2.9 Checking Bias](#29-checking-bias)


## 1. Introduction to Experiment Design and Recommendation Engines

In a nutshell:

- Experiments: A/B tests = hypothesis tests, ANOVAs, etc.
- Recommendation engines: suggesting items a user might like.

Experiments can be used to evaluate the quality of the recommendations.

Experiments are broken into 2 groups:

- Treatment
- Control

How can we determine when an experiment has completed?

- Define metrics beforehand.
- Choose 1-2 metrics and their threshold values, which are the flags.
- Make sure we collect enough data to make a recommendation.

How to decided what to test?

- Often marketing decides according to their priorities.
- We need to consider the impact of the test in the operations.

In every new experiment, you need to consider

- All data sources and all data that is incoming: shops, inventories, etc.
- How can be measured whether an experiment worked or not.

Most experiments fail: the outcome is there is no significant difference between the treatment and the control group. That is normal, because we're trying to discover things that are not so obvious; if they were easy, why aren't we using them?

When creating the control/treatment groups:

- Be mindful of the time of year if the business has seasonal patterns.
- Consider the users' past behavior: maybe they have significant patterns that affect the group result.

## 2. Concepts in Experiment Design

### 2.1 What Is an Experiment?

Experiments are run to check hypotheses; however, we need to take into account that **correlation does not imply causation**.

Example: we have an online store and we want to check which UX leads to more purchases: (i) clicking on a product opens a new product tab or (ii) clicking on a product opens an overlay. To perform the experiment which answers the question:

1. We need to compare two groups: one with treatment (i) or control, one for treatment (ii) or overlay.
2. We need to make sure that the only difference between the groups is the feature in question, i.e., we need to control that there are no other differences. One way to achieve that is to do it randomly.

![Experiment Design](./pics/experiment_design.jpg)

However, often it's not possible to run a true experiment, which requires at least two groups of users randomly selected, one of the groups being the control group. We can say there's a spectrum of experiments:

- **Experiments** (one extreme): two groups, randomly generated, one of them control; we have full control over the features. This is typical in medical sciences.
- **Observational studies** (the other extreme): not possible to have two random groups, so we don't have control over the features. Sometimes, the reason for having an observational study are ethical issues. We cannot infer any causation, but we can use them to understand dynamics and formulate hypotheses to be tested.
- **Quasi-experiments** (between the extremes): we have some control over the features; e.g., we implement a new feature and test it without control group, or the group with the new feature is not random. Depending on the product, this can be quite common: for instance, if we launch a feature in *beta* and customers try it, the group is not random anymore!

Lecture videos:

- [What Is An Experiment](https://www.youtube.com/watch?v=fH_xF5_SDCE&t=106s)
- [What Is An Experiment Pt 2](https://www.youtube.com/watch?v=PYzN1usi7QY&t=185s)

### 2.2 Types of Experiment

The two most important typed of experiments are:

- **Between**: each group A/B tries one treatment control/experiment.
- **Within**: each group A/B tries both treatments; the advantage of this type of experiment is that we can account for the variance introduced by the subjects. However, not always is possible to design a *within* study.

![Between vs. Within Experiments](./pics/between_within_experiments.jpg)

Another type of experiments are **factorial**: we test several factors, not only one; these lead to ANOVA analyses and require a stricter control.

Lecture video: [Types Of Experiments](https://www.youtube.com/watch?v=7ihDj4M7EiU&t=190s)

### 2.3 SMART Experiments

Experiments should be designed in a SMART way:

> - Specific: Make sure the goals of your experiment are specific.
> - Measurable: Outcomes must be measurable using objective metrics
> - Achievable: The steps taken for the experiment and the goals must be realistic.
> - Relevant: The experiment needs to have a purpose behind it.
> - Timely: Results must be obtainable in a reasonable time frame.

### 2.4 Types of Sampling

The most common way of random sampling is **Simple Random Sampling**: we have a population and give each individual an equal chance of being selected. However, sometimes (often) that is not completely possible. For instance, we might have a population divided in different living regions (urban 50%, suburban 30%, rural 20%), so some regions are unrepresented. In those cases, instead of choosing randomly from the total population, we allocate a given amount of people to each region relative to the percentage of people living there; that way, we assure a representative amount in each region instead of leaving the selection completely to chance. That is called **Stratified Random Sampling**.

![Sampling](./pics/sampling.jpg)

Both Simple Random Sampling and Stratified Random Sampling are **probabilistic sampling methods**; however, there exist also **non-probabilistic sampling methods**, such as **Convenience Sampling**: record information from available units, e.g., college students in a university study. These sometimes are the only way of of conducting the experiment, but they might lead to false results, because they use non-representative samplings.

Lecture video: [Types of Sampling](https://www.youtube.com/watch?v=GF_eQqNoarI&t=1s).

### 2.5 Measuring Outcomes

How can we measure the effectiveness of a video recommendation engine? We need to define measurable **evaluation metrics** beforehand, e.g.:

- Video watch time: the longer, the better.
- Video ranking: the higher, the better.
- Number of search queries after watching the video: the more the better, because the video might have arisen curiosity.

However, **those metrics might be misleading**:

- The engine suggests only longer videos, not better ones.
- The engine suggests videos which have a high ranking, ignoring the ones with few rankings.
- The engine suggests videos which originate many queries, but because they are confusing or not good enough.

So what should we do? Consider goals of the study separate from the metrics: the metric might be a proxy to what is being measured. Additionally, consider all implications of a metric value.

Lecture videos:

- [Measuring Outcomes Pt 1](https://www.youtube.com/watch?v=HPmMEkbT2uE&t=5s)
- [Measuring Outcomes Pt 2](https://www.youtube.com/watch?v=yLdXcRXcfPw&t=2s)

### 2.6 Creating Metrics

Going back to the online store example, we need to find a way to divide the two groups; notes:

- The flow of steps the user follows from start to end are called **user funnel**, because we loose users from step to step.
- Two groups mean two different funnels.
- In the case of an online store, we can implement the separation with a cookie in the beginning: when the user opens the page/starts a session, a cookie which randomly assigns the user to groups A/B is created, which is permanently stored for that user. The cookie is **unit of diversion** and it should be unnoticeable for the user; depending on the experiment, we might have different types of diversions:
  - Event-based: when opening page.
  - Cookie-based: a user has one permanent cookie.
  - Account-based: every logged user has a group assigned.

We need to track two kinds of metrics:

1. Evaluation metrics: metrics we expect to change due to the treatment. We compare the groups with them.
2. Invariant metrics: metrics we expect to be constant; we measure them to make sure that the groups are the same, treatment aside.

![Metrics](./pics/metrics.jpg)

Lecture video: [Creating Metrics](https://www.youtube.com/watch?v=__7tzDUY870&t=7s)

### 2.7 Controlling Variables

As mentioned before, *correlation does not mean causation*. Sometimes variables are correlated, but that relationship can be:

- by chance
- or caused by a **confounding variable**: a confounding variable is a hidden factor which influences both correlated variables.

Example: ice cream consumption and crime rates are correlated; the confounding variable can be the temperature: the hotter it is, the more ice cream people eat and more crimes occur, without both being related more than by the temperature.

If we want to argue causality, we need to change only one factor/variable and control the experiment very thoroughly.

Lecture video: [Controlling Variables](https://www.youtube.com/watch?v=pLTneSg2MRY)

### 2.8 Checking Validity

Validity is related to how well the conclusions of the experiment can be supported.

There are 3 types of validity measures:

1. Construct validity: degree to which an experiment's metric result supports the goals of the study; a bad construct validity example is the number of search queries in the online store example.
2. Internal validity: degree to which a causality claim can be supported. If we have 2 correlated variables, but we don't account for any other variables, the causation is not well supported; we need to consider other variables to state causation.
3. External validity: how generalizable the results are. This is related to the representativeness of the sample: the more representative, the more generalizable.

Lecture video: [Checking Validity](https://www.youtube.com/watch?v=H3H1SZXqDmQ&t=2s).

### 2.9 Checking Bias



Lecture video: [Checking Bias](https://www.youtube.com/watch?v=ppjNNY4DhPw&t=1s).