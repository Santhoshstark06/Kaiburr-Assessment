#Explanatory Data Analysis and Feature Engineering

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the consumer complaint dataset into a pandas dataframe
df = pd.read_csv('consumer_complaints.csv')

# EDA
# Plot the distribution of complaint types
plt.figure(figsize=(12,6))
sns.countplot(x='Product', data=df)
plt.title('Distribution of Complaint Types')
plt.xlabel('Product')
plt.ylabel('Count')
plt.show()

# Plot the distribution of complaint lengths
df['complaint_length'] = df['Consumer complaint narrative'].str.len()
plt.figure(figsize=(12,6))
sns.histplot(df, x='complaint_length', hue='Product', multiple='stack', bins=50)
plt.title('Distribution of Complaint Lengths by Product')
plt.xlabel('Complaint Length')
plt.ylabel('Count')
plt.show()

# Feature Engineering
# Tokenize the consumer complaints
df['tokens'] = df['Consumer complaint narrative'].apply(word_tokenize)

# Remove stop words from the consumer complaints
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# Create a TF-IDF matrix from the tokenized and cleaned consumer complaints
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['Consumer complaint narrative'])

#Text Pre-Processing

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Load the consumer complaint dataset into a pandas dataframe
df = pd.read_csv('consumer_complaints.csv')

# Define a function to pre-process the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the tokens back into a string
    text = ' '.join(tokens)
    return text

# Apply the pre-processing function to the consumer complaint text
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(preprocess_text)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load the consumer complaint dataset into a pandas dataframe
df = pd.read_csv('consumer_complaints.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Consumer complaint narrative'], df['Product'], test_size=0.2, random_state=42)

# Create a pipeline for feature extraction and model training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LinearSVC())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load the consumer complaint dataset into a pandas dataframe
df = pd.read_csv('consumer_complaints.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Consumer complaint narrative'], df['Product'], test_size=0.2, random_state=42)

# Define a list of pipelines for different models
pipelines = [
    Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LinearSVC())
    ]),
    Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', MultinomialNB())
    ]),
    Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])
]

# Fit each pipeline to the training data and evaluate on the test data
for pipeline in pipelines:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print('Model performance:')
    print(classification_report(y_test, y_pred))

