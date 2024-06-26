# -*- coding: utf-8 -*-
"""SentimentAnalysis-AmazonReviews.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GrdiKkM3aPumONqddQLVZ48eSnM4S5oY

This code snippet first ensures that the NLTK (Natural Language Toolkit) and spaCy libraries are installed, which are commonly used for natural language processing tasks. Then, it calculates the term frequency (tf) of each word in the "reviewText" column of the DataFrame. This is achieved by splitting each review text into individual words and tallying the occurrence of each word across all reviews. The resulting DataFrame contains two columns: "words" and "tf", representing unique words and their corresponding term frequencies, respectively. This analysis provides valuable insights into the distribution and prevalence of words within the dataset, aiding in further exploration and understanding of the textual data.
"""

!pip install nltk
!pip install spacy
!tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.head()

"""This code reads a CSV file containing Amazon reviews data into a DataFrame using the Pandas library. It assumes the file is named "amazon_reviews.csv" and is located in the "/content/" directory. After reading the file, it displays the first few rows of the DataFrame to provide a preview of the data."""

import pandas as pd

# Assuming your file is named "amazon_reviews.csv" and it's located in the "/content/" directory
file_path = "/content/amazon_reviews.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
df.head()

"""To count the number of instance"""

# computing number of rows
rows = len(df.axes[0])

# computing number of columns
cols = len(df.axes[1])

print("Number of Rows: ", rows)
print("Number of Columns: ", cols)

"""df.describe()*italicized text*"""

df.describe()

"""This code snippet converts all text in the 'reviewText' column of the DataFrame to lowercase using the `.str.lower()` method in Pandas. By doing this, it ensures that the text is standardized to lowercase, which can be helpful for subsequent analysis such as text processing or sentiment analysis. The `.head()` function then displays the first few rows of the DataFrame with the updated 'reviewText' column."""

df['reviewTime'] = pd.to_datetime(df['reviewTime'])

# Extract year and month from 'reviewTime' to group by
df['reviewYearMonth'] = df['reviewTime'].dt.to_period('M')

# Count the number of reviews per year-month
reviews_per_month = df.groupby('reviewYearMonth').size()

# Plotting the time series
plt.figure(figsize=(10, 6))
reviews_per_month.plot()
plt.title('Time Series of Number of Reviews')
plt.xlabel('Review Year-Month')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

import seaborn as sns

numeric_cols = ['overall', 'helpful_yes', 'day_diff']

# Creating a correlation matrix
corr_matrix = df[numeric_cols].corr()

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Columns')
plt.show()

df['reviewText'] = df['reviewText'].str.lower()
df.head()

"""This code removes any non-alphanumeric characters and whitespace from the 'reviewText' column in the DataFrame. Then, it displays the first few rows of the DataFrame with the cleaned 'reviewText'."""

df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')
df.head()

"""This code removes any punctuation characters from the 'reviewText' column in the DataFrame. Then, it displays the first 20 entries of the cleaned 'reviewText' column."""

#Punctuation
df["reviewText"] = [re.sub("[^a-zA-Z]", " ", i) for i in df["reviewText"].astype(str)]
df["reviewText"].head(20)

"""This code removes any digits from the 'reviewText' column in the DataFrame. Then, it displays the first 20 entries of the cleaned 'reviewText' column."""

df['reviewText'] = df['reviewText'].str.replace('\d', '')
df.head(20)

"""This code downloads the NLTK stopwords corpus and prints out the list of English stopwords. Then, it removes the stopwords from the 'reviewText' column in the DataFrame 'df' and displays the last 20 entries of the cleaned 'reviewText' column."""

import nltk
nltk.download('stopwords')

# Get the list of stopwords
from nltk.corpus import stopwords
sw = stopwords.words('english')
print(sw)


# Remove stopwords from the 'reviewText' column
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
print(df["reviewText"].tail(20))

"""This code snippet utilizes the `re` module to remove digits from the 'reviewText' column in the DataFrame 'df'. It applies a regular expression `\d` to replace any digit characters with an empty string. Finally, it prints the last 20 entries of the modified 'reviewText' column."""

import re



# Remove digits from the 'reviewText' column
df["reviewText"] = df["reviewText"].apply(lambda x: re.sub("\d", "", str(x)))

# Print the last few rows of the modified 'reviewText' column
print(df["reviewText"].tail(20))

"""This code removes infrequent words (those that appear only once) from the 'reviewText' column of the DataFrame 'df'. First, it creates a temporary DataFrame `temp_df` containing the word frequency counts using `value_counts()`. Then, it identifies words with a count of 1 or less and stores them in the `drops` variable. Next, it applies a lambda function to filter out these infrequent words from each entry in the 'reviewText' column, using a list comprehension and the `join()` and `split()` methods. Finally, it prints the last 20 entries of the modified 'reviewText' column."""

temp_df = pd.Series(" ".join(df["reviewText"]).split()).value_counts()
print(temp_df)

drops = temp_df[temp_df <=1]
print(drops)
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

"""This code snippet first downloads the NLTK tokenizer data required for word tokenization using `nltk.download("punkt")`. Then, it tokenizes the text in the 'reviewText' column using the `TextBlob` library, which is part of the NLTK package. It converts each text entry into a list of words. The resulting DataFrame contains these tokenized words in the 'reviewText' column. However, it seems there's a missing operation in the code, as it's not clear what you want to do after tokenization. If you have any specific tasks or analysis you'd like to perform with this tokenized data, please specify."""

nltk.download("punkt")

df["reviewText"].apply(lambda x: TextBlob(x).words).head(20)
df['reviewText']

"""This code segment utilizes the SpaCy library to perform lemmatization on the text in the "reviewText" column of the DataFrame. First, it loads the English language model using `spacy.load("en_core_web_sm")`. Then, it defines a function called `lemmatize_text()` that takes a text input, processes it using SpaCy's NLP pipeline, and extracts lemmas (base forms) of tokens. Finally, it applies this lemmatization function to each entry in the "reviewText" column of the DataFrame using the `apply()` function. The resulting DataFrame contains the lemmatized text in the "reviewText" column, with the entire text displayed for each entry due to setting the display options."""

import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define a function to lemmatize text
def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

# Apply the lemmatize_text function to the "reviewText" column of your DataFrame
df["reviewText"] = df["reviewText"].apply(lemmatize_text)
# Set display options to show the entire text
pd.set_option('display.max_colwidth', None)

# Print the entire "reviewText" column along with the first 20 rows
print(df["reviewText"])

"""This code defines a dictionary called `contraction_mapping` that contains common contractions as keys and their expanded forms as values. Then, it defines a function called `expand_contractions()` that takes a text input and expands any contractions found in the text using the provided mapping. The function uses a regular expression pattern to find contractions in the text and replaces them with their expanded forms. Finally, the function is applied to the "reviewText" column of the DataFrame using the `apply()` function, and the expanded text is printed for the first 20 rows."""

contraction_mapping = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
}

# Function to expand contractions
def expand_contractions(text, contraction_mapping):
    # Regular expression pattern to find contractions
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)

    # Function to replace contractions with their expansions
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    # Replace contractions in text using the pattern and the function to replace
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

# Apply the expand_contractions function to the "reviewText" column of your DataFrame
df["reviewText"] = df["reviewText"].apply(lambda x: expand_contractions(x, contraction_mapping))

# Print the entire "reviewText" column along with the first 20 rows
pd.set_option('display.max_colwidth', None)
print(df["reviewText"].head(20))

"""This code defines a function called `normalize_text()` that takes a text input and normalizes URLs and email addresses by replacing them with tokens ('URL' and 'EMAIL' respectively). The function uses regular expressions to identify URLs and email addresses in the text and replaces them with the specified tokens. Then, the function is applied to the "reviewText" column of the DataFrame using the `apply()` function. Finally, the normalized text is printed for the entire "reviewText" column, along with the first 20 rows."""

import re

# Function to normalize URLs, emails, etc.
def normalize_text(text):
    # Replace URLs with a token
    text = re.sub(r'http\S+', 'URL', text)
    # Replace email addresses with a token
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', 'EMAIL', text)
    # Add more normalization rules as needed

    return text

# Apply the normalize_text function to the "reviewText" column of your DataFrame
df["reviewText"] = df["reviewText"].apply(normalize_text)

# Print the entire "reviewText" column along with the first 20 rows
pd.set_option('display.max_colwidth', None)
print(df["reviewText"])

"""This code first prints the current columns in the DataFrame. Then, it specifies a list of columns to remove, including 'unixReviewTime', 'reviewerName', 'asin', 'day_diff', and 'reviewerID'. After that, it drops these specified columns from the DataFrame using the `drop()` function with the `columns` parameter. Finally, it prints the remaining columns in the DataFrame and displays the top 20 rows to verify the changes."""

# Print the current columns in the DataFrame
print("Current Columns:")
print(df.columns)

# List of columns to remove
columns_to_remove = ['unixReviewTime', 'reviewerName', 'asin', 'day_diff','reviewerID']

# Drop the specified columns from the DataFrame
df.drop(columns=columns_to_remove, inplace=True)

# Print the remaining columns
remaining_columns = df.columns
print("\nRemaining Columns:")
print(remaining_columns)

# Display the top 20 rows of the DataFrame
print("\nTop 20 Rows:")
print(df.head(20))

"""This code utilizes the `tabulate` library to print the remaining columns of the DataFrame in a table format. It first creates a DataFrame containing the remaining columns, then prints the table using the `tabulate` function with the specified headers and table format. This format makes it easier to visualize the remaining columns."""

from tabulate import tabulate

# Print the remaining columns in a table format
print("Remaining Columns:")
print(tabulate(pd.DataFrame(df.columns, columns=['Columns']), headers='keys', tablefmt='psql'))

# Print 10 entries from the DataFrame
print("Top 10 Entries:")
print(tabulate(df.head(10), headers='keys', tablefmt='psql'))

"""This code recalculates the helpfulness ratio by dividing the 'helpful_yes' column by the 'total_vote' column and then prints the correlation between 'helpful_yes' and 'total_vote'. After that, it visualizes the distribution of the helpfulness ratio using a histogram with a kernel density estimation (KDE) plot overlaid. The histogram shows the frequency distribution of the helpfulness ratio values, while the KDE plot provides a smooth estimate of the probability density function. This visualization helps to understand how helpful the reviews are relative to the total number of votes they received."""

# Recalculate the helpfulness ratio
df['helpfulness_ratio'] = df['helpful_yes'] / df['total_vote']

# Print the correlation between 'helpful_yes' and 'total_vote'
correlation = df['helpful_yes'].corr(df['total_vote'])
print("Correlation between helpful_yes and total_vote:", correlation)

# Visualize the distribution of helpfulness ratio
plt.figure(figsize=(8, 6))
sns.histplot(df['helpfulness_ratio'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Helpfulness Ratio')
plt.xlabel('Helpfulness Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

"""This code calculates the correlation coefficient between the 'helpful_yes' and 'total_vote' columns in the DataFrame 'df', which indicates the strength and direction of their linear relationship. The correlation coefficient ranges from -1 to 1, where:

- 1 indicates a perfect positive correlation,
- -1 indicates a perfect negative correlation, and
- 0 indicates no correlation.

The printed output shows the correlation coefficient value between the two columns.
"""

import pandas as pd

# Assuming 'helpful_yes' and 'total_vote' columns are present in your DataFrame 'df'

# Calculate the correlation between 'helpful_yes' and 'total_vote'
correlation = df['helpful_yes'].corr(df['total_vote'])

# Print the correlation coefficient
print("Correlation between 'helpful_yes' and 'total_vote':", correlation)

"""This code performs a time series analysis of the mean overall ratings over months for each year present in the DataFrame 'df'. It first converts the 'reviewTime' column to datetime format and extracts the year and month from it. Then, it groups the data by year and month, calculating the mean overall rating for each month. Finally, it plots the time series of mean overall ratings over months within each year, showing how the ratings change over time. Each year's data is represented by a separate line plot with markers indicating individual data points. The plot includes labels, a title, and a legend to aid interpretation."""

import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'overall' and 'reviewTime' columns are present in your DataFrame 'df'

# Convert 'reviewTime' column to datetime format
df['reviewTime'] = pd.to_datetime(df['reviewTime'])

# Extract year and month from 'reviewTime' column
df['year'] = df['reviewTime'].dt.year
df['month'] = df['reviewTime'].dt.month

# Group the data by year and month and calculate the mean overall rating for each month
overall_ratings_monthly = df.groupby(['year', 'month'])['overall'].mean()

# Plot the time series of mean overall ratings over months within each year
plt.figure(figsize=(12, 6))
for year in overall_ratings_monthly.index.levels[0]:
    data = overall_ratings_monthly.loc[year]
    plt.plot(data.index, data.values, marker='o', label=year)

plt.title('Mean Overall Ratings Over Months')
plt.xlabel('Month')
plt.ylabel('Mean Overall Rating')
plt.legend(title='Year')
plt.grid(True)
plt.xticks(range(1, 13))
plt.show()

"""This code defines a function `extract_first_value` that extracts the first value from a list represented as a string. It then applies this function to each value in the 'helpful' column of the DataFrame 'df', converting the string representation into an integer and summing up the results. Finally, it prints the total number of helpful votes. If there's any error encountered during conversion, it handles it gracefully by returning 0."""

# Function to extract the first value from a list in a string representation
def extract_first_value(string_list):
    try:
        return int(string_list.strip('[]').split(',')[0])
    except ValueError:
        return 0

# Apply the function to each value in the 'helpful' column and sum up the results
total_helpful_votes = df['helpful'].apply(extract_first_value).sum()

print("Total Helpful Votes:", total_helpful_votes)

"""This code defines a function `extract_second_value` that extracts the second value from a list represented as a string. It then applies this function to each value in the 'helpful' column of the DataFrame 'df', converting the string representation into an integer and summing up the results. Finally, it prints the total number of unhelpful votes. If there's any error encountered during conversion, it handles it gracefully by returning 0."""

# Function to extract the second value from a list in a string representation
def extract_second_value(string_list):
    try:
        return int(string_list.strip('[]').split(',')[1])
    except ValueError:
        return 0

# Apply the function to each value in the 'helpful' column and sum up the results
total_unhelpful_votes = df['helpful'].apply(extract_second_value).sum()

print("Total Unhelpful Votes:", total_unhelpful_votes)

"""This code creates a pie chart to visualize the distribution of helpful and unhelpful votes. It sets up the data with the total counts of helpful and unhelpful votes, defines the labels for the pie chart, and specifies colors for each section. Then, it creates the pie chart using Matplotlib, with percentages displayed on each section and a title indicating the distribution of votes. Finally, it displays the pie chart. If any of the sizes are negative, it prints an error message indicating that sizes must be non-negative."""

import matplotlib.pyplot as plt

# Total helpful and unhelpful votes
total_helpful_votes = 6444
total_unhelpful_votes = 7478

# Data for the pie chart
labels = ['Helpful', 'Unhelpful']
sizes = [total_helpful_votes, total_unhelpful_votes]

# Check if sizes are non-negative
if any(size < 0 for size in sizes):
    print("Error: Sizes must be non-negative.")
else:
    # Set up the colors for the pie chart
    colors = ['#ff9999', '#66b3ff']

    # Create the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Helpful and Unhelpful Votes')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

"""This code calculates the number of unique words in the DataFrame's "words" column using the `nunique()` function. It returns the count of unique words present in the column."""

tf["words"].nunique()

"""This code computes the descriptive statistics for the "tf" column of the DataFrame `tf`. It includes various percentiles such as 5th, 10th, 25th, 50th (median), 75th, 80th, 90th, 95th, and 99th percentiles. The `.T` at the end transposes the result, making it easier to read."""

tf["tf"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

"""This code generates a bar plot using the DataFrame `tf` filtered for words with a term frequency greater than 500. It plots the selected words on the x-axis and their corresponding term frequencies on the y-axis. Finally, it displays the bar plot using `plt.show()`. This visualization helps identify the most frequent words in the dataset."""

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

"""This code creates a word cloud visualization using the review text data from the DataFrame `df`. It concatenates all the review texts into a single string variable named `text`. Then, it generates a word cloud using the `WordCloud` class and the `generate()` method, passing the `text` variable as input. Finally, it displays the word cloud using `plt.imshow()` and sets the axis off to remove axis labels, followed by `plt.show()` to show the word cloud plot. Word clouds are useful for visualizing the most frequent words in a text corpus."""

text = " ".join(i for i in df.reviewText)
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

"""This code generates a word cloud with specific settings. It sets the maximum font size to 50 using the `max_font_size` parameter, limits the maximum number of words in the cloud to 100 with the `max_words` parameter, and sets the background color to white using `background_color="white"`. Then, it creates the word cloud using the `WordCloud` class with the specified settings and generates it from the `text` variable. Finally, it displays the word cloud plot with `plt.imshow()` and hides the axis labels with `plt.axis("off")`, followed by `plt.show()` to show the plot. This customization allows for more control over the appearance of the word cloud."""

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

"""Using Random forest Classifier"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Split data into features (X) and target variable (y)
X = df['reviewText']
y = df['overall']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create and display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)



"""Using logistic regression"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x=df.drop(columns='overall',axis=1)
y=df['overall']
#separate the data and label
x=df['reviewText'].values
y=df['overall'].values


#convert text to numerical data
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['reviewText'].values.astype('U'))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=2)

model=LogisticRegression()
model.fit(x_train,y_train)
#finding the accuracy score
#on train data
x_train_data=model.predict(x_train)
train_data_accuracy=accuracy_score(x_train_data,y_train)
print('Accuracy of training data: ', train_data_accuracy)

"""training on test data"""

x_test_data=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_data,y_test)
print('Accuracy of testing data: ',test_data_accuracy)

"""Using confusion matrix to show incorrect and correct pridiction."""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Make predictions on the test data
y_pred = model.predict(x_test)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)


# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
           xticklabels=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
            yticklabels=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

