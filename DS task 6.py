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

