import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset

df = pd.read_csv('preprocessed_data.csv')
pd.set_option('display.max_columns', None)

# Assuming 'data' is your DataFrame with features and target
X = df.drop('is_promoted', axis=1)
y = df['is_promoted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)






