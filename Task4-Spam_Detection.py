import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv("spam.csv")
print(df.head())

# Create a new column 'spam' based on the 'Category' column
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
print(df.head())

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], test_size=0.25)

# Convert text data into a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_Count = v.fit_transform(X_train.values)
print(X_train_Count.toarray()[:3])

# Train a Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_Count, y_train)

# Example predictions
emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20 percent discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
print(model.predict(emails_count))

# Measure accuracy of the model
X_test_Count = v.transform(X_test)
print(model.score(X_test_Count, y_test))
