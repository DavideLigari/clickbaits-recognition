# Clickbait Recognition

This repository contains a project aimed at building a classifier to distinguish between clickbait headlines and regular headlines. Clickbait headlines are designed to attract attention by using deceptive, sensationalized, or misleading content.

## Problem Definition and Data

The objective is to analyze a dataset of headlines and build a classifier that can predict whether a given headline is a clickbait or not. 
The dataset consists of 32,000 headlines, evenly divided into two classes: 'clickbait' and 'non-clickbait'. 
It includes three sets: training, validation, and test sets, with 24,000, 4,000, and 4,000 samples, respectively.
The headlines are stored in text files, with each line representing a single headline.

## Assignment

To complete the programming assignment, the following tasks are expected:

1. **Data Analysis**: Analyze and comment on the characteristics of the dataset.
2. **Data Pre-processing**: Design and implement a suitable data pre-processing procedure to prepare the headlines for classification.
3. **Classification Model**: Implement, train, and evaluate one or more classification models to predict clickbait headlines. Use the Python programming language and any machine learning library (including pvml) for implementation.
4. **Model Evaluation and Visualization**: Utilize appropriate data processing and visualization techniques to analyze the behavior of the trained models.

Two scenarios should be considered for the evaluation: a generic scenario where all errors are equally important, and a 'precision-oriented' scenario where minimizing the chance of false positives is crucial.

## Repository Structure

The repository is structured as follows:

```
|- data/
|  |- accuracy_analysis/
|  |- fpr_analysis/
|  |- stopwords.txt
|
|- dataset/
|- code/
```
The `code` directory contains the notebook used to conduct the analysis and the python scripts, which contains some customized functions 

The `data` contains the results of the analisis performed. 

The `dataset` directory contains the train,validation and test sets for both classes 




