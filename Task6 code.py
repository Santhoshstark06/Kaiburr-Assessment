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
