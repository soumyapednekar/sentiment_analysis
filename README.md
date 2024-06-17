# Sentiment Analysis

This repository contains code and resources for conducting sentiment analysis on Amazon product reviews using machine learning algorithms such as Logistic Regression and Random Forest Classifier. 



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
- Several machine learning models were trained and evaluated:

1. Logistic Regression: Demonstrated balanced performance across classes with around 70% accuracy, but some difficulty classifying negative sentiments.

2. Linear SVM: Achieved around 71% accuracy, effectively separating sentiment classes and performing best on positive sentiments. 

3. Decision Tree: Moderate performance at 68% accuracy, with issues in generalization and classifying negative sentiments indicating overfitting.

4. Random Forest: The best performing model at 69% accuracy without cross-validation and 74% with cross-validation. Showed robust and balanced performance across classes due to its ensemble nature reducing overfitting.

Cross-validation using 5 folds was implemented to evaluate model generalization. Logistic Regression and Linear SVM maintained stable cross-validation scores, while Random Forest improved significantly after cross-validation, confirming its superiority.

 ### 3.Prediction Results and Analysis:
- The prediction models provided insights into the sentiment of product reviews.
- Analyzed the accuracy and performance of the models to determine their effectiveness in predicting overall ratings.
-The accuracy of logistic regression on the training data was approximately 82.33%, while on the test data, it achieved an accuracy of approximately 81.15%. This indicates that the model generalizes reasonably well to unseen data.
-Confusion matrix analysis provided insights into the model's performance across different rating classes, highlighting areas of accuracy and potential misclassifications.

### 4.Data Preprocessing and Prediction:
-Predictions of overall ratings were produced by integrating the preprocessed textual data into the prediction algorithms.
- Assessed the combined strategy to determine how it affected the dependability and accuracy of the predictions.

### 5.Experimental Insights: 
1) Random Forest emerged as the most robust and reliable model for sentiment classification of product reviews by effectively managing complexities in natural language data.

2) Ensemble techniques like Random Forest tend to perform better than individual models for this task.

3) Careful model selection, validation, and hyperparameter tuning are crucial for achieving reliable sentiment predictions.

- The project demonstrated the potential of applying machine learning for extracting actionable consumer insights from online reviews to drive product enhancements, targeted marketing, and improved customer experiences in e-commerce.
 Future extensions were suggested like multilingual analysis, aspect-based sentiment analysis, real-time monitoring, and exploring deep learning models.

 -The textual data was successfully cleaned and standardized by the pre-processing approaches, which enhanced the analysis.
- We could draw some conclusions and insights from the visualization and time series analysis.
- Challenges included addressing class imbalances and refining feature selection techniques, which required further exploration and experimentation.

