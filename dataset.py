import pandas as pd
import matplotlib.pyplot as plt

# Load dataset

df = pd.read_csv('train_LZdllcl.csv')
pd.set_option('display.max_columns', None)

# Handling missing values

# Replace null values with the mode

df['education']=df['education'].fillna(df['education'].mode()[0])

# Replace null values with the median

df['previous_year_rating']=df['previous_year_rating'].fillna(df['previous_year_rating'].median())











