# Clickbait Recognition

This repository contains a project aimed at building a classifier to distinguish between clickbait headlines and regular headlines. Clickbait headlines are designed to attract attention by using deceptive, sensationalized, or misleading content.

## Problem Definition and Data

The objective of this project is to analyze a dataset of headlines and develop a classifier that can predict whether a given headline is a clickbait or not. The dataset consists of 32,000 headlines, evenly divided into two classes: 'clickbait' and 'non-clickbait'. It includes three sets: training, validation, and test sets, with 24,000, 4,000, and 4,000 samples, respectively. The headlines are stored in text files, with each line representing a single headline.

## Tasks

To achieve the project goal, the following tasks were performed:

1. **Data Analysis**: Analyzed the dataset and provided insights into its characteristics.
2. **Data Pre-processing**: Designed and implemented an effective data pre-processing procedure to prepare the headlines for classification.
3. **Classification Model**: Logistic regression and multinomial Naive Bayes classifier were trained and compared in terms of accuracy and false positive rate. 
5. **Model Evaluation and Visualization**: Utilized suitable data processing and visualization techniques to analyze the behavior of the trained models.

Two evaluation scenarios were considered: a generic scenario where all errors are equally important, and a 'precision-oriented' scenario where minimizing false positives was prioritized.

## Repository Structure

The repository is organized as follows:

```
|- data/
|  |- accuracy_analysis/
|  |- fpr_analysis/
|  |- stopwords.txt
|
|- dataset/
|- code/
```

The `code` directory contains a notebook used for analysis and Python scripts that include customized functions for the project.

The `data` directory stores the results of the performed analysis.

The `dataset` directory provides the training, validation, and test sets for both clickbait and non-clickbait classes.

## Contact
If you have any questions or suggestions regarding this project, please feel free to contact me via email at davide.ligari01@gmail.com

Thank you for your interest in this project!
