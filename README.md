# Sentiment Analysis

This repository contains code and resources for conducting sentiment analysis on Amazon product reviews using machine learning algorithms such as Logistic Regression and Random Forest Classifier. The project is part of a midterm project report for Group 04, consisting of Hitakshi Tanna, Rudraja Vansutre, Somita Chaudhari, Soumya Pednekar, and Tejashree Betgar.



## Background

Theoretical foundations necessary for understanding the methodology include text preprocessing techniques such as lowercasing, tokenization, stopword removal, and TF-IDF vectorization. The Random Forest Classifier, an ensemble learning method, is utilized for classification tasks, along with key evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix analysis.

## Data Description

The dataset contains information on Amazon product reviews with columns such as `asin`, `reviewerID`, `reviewerName`, `helpful`, `reviewText`, `overall`, `summary`, `unixReviewTime`, `reviewTime`, `day_diff`, `helpful_yes`, and `total_vote`. Statistical measures, data visualizations (heatmaps, time series plots, pie charts), and correlation analyses were performed to understand the dataset better.

## Data Analysis and Preprocessing

- Text preprocessing involved steps like lowercase conversion, punctuation removal, and stopword elimination.
- TF-IDF vectorization was applied to transform text data into numerical features.
- The dataset was split into training and testing sets for model evaluation.

## Methodology

- **Logistic Regression:** Used for binary classification tasks based on text features.
- **Random Forest Classifier:** Employed for ensemble learning and classification of reviews into sentiment categories.

## Results and Analysis
### 1.Data Analysis:
- Pie charts and bar plots were used to visualize the distribution of the helpfulness ratio and the overall number of votes.
- Performed a time series analysis to investigate how each year's mean overall ratings changed over time.
- carried out a correlation analysis to look at the connection between the overall number of votes and the helpful votes.

### 2.Prediction using Data Mining Models/Methods:
- We used Random Forest and Logistic Regression algorithms for predicting overall ratings based on review texts.

 ### 3.Prediction Results and Analysis:
- The prediction models provided insights into the sentiment of product reviews.
- Analyzed the accuracy and performance of the models to determine their effectiveness in predicting overall ratings.
-The accuracy of logistic regression on the training data was approximately 82.33%, while on the test data, it achieved an accuracy of approximately 81.15%. This indicates that the model generalizes reasonably well to unseen data.
-Confusion matrix analysis provided insights into the model's performance across different rating classes, highlighting areas of accuracy and potential misclassifications.

### 4.Data Preprocessing and Prediction:
-Predictions of overall ratings were produced by integrating the preprocessed textual data into the prediction algorithms.
- Assessed the combined strategy to determine how it affected the dependability and accuracy of the predictions.

### 5.Experimental Insights: -
 -The textual data was successfully cleaned and standardized by the pre-processing approaches, which enhanced the analysis.
- We could draw some conclusions and insights from the visualization and time series analysis.
- Challenges included addressing class imbalances and refining feature selection techniques, which required further exploration and experimentation.

