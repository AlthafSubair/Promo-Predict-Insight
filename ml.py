import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset

df = pd.read_csv('preprocessed_data.csv')
pd.set_option('display.max_columns', None)

print(df.head())




