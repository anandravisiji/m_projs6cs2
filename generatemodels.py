import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pickle

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
data = pd.read_csv("Coursera.csv")

# Data preprocessing
data = data[['Course Name', 'Difficulty Level', 'Course Description', 'Skills']]

# Replace certain characters and remove unnecessary spaces
for column in ['Course Name', 'Course Description']:
    data[column] = data[column].str.replace(' ', ',')
    data[column] = data[column].str.replace(',,', ',')
    data[column] = data[column].str.replace(':', '')
    data[column] = data[column].str.replace('_', '')

# Removing parentheses from 'Skills' column
data['Skills'] = data['Skills'].str.replace('(', '')
data['Skills'] = data['Skills'].str.replace(')', '')

# Create a 'tags' column
data['tags'] = data['Course Name'] + ' ' + data['Difficulty Level'] + ' ' + data['Course Description'] + ' ' + data['Skills']

# Lowercasing and stemming
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

data['tags'] = data['tags'].apply(lambda x: x.lower())
data['tags'] = data['tags'].apply(stem)

# Create new dataframe with relevant columns
new_df = data[['Course Name', 'tags']]
new_df.rename(columns = {'Course Name': 'course_name'}, inplace = True)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute the cosine similarity matrix
similarity = cosine_similarity(vectors)

# Save the models to disk
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(new_df, open('courses.pkl', 'wb'))

print("Models have been saved!")
