import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize


# Load dataset

df = pd.read_csv('train_LZdllcl.csv')
pd.set_option('display.max_columns', None)

# Handling missing values

# Replace null values with the mode

df['education']=df['education'].fillna(df['education'].mode()[0])

# Replace null values with the median

df['previous_year_rating']=df['previous_year_rating'].fillna(df['previous_year_rating'].median())

# No duplicate entries found

# Droped "employee_id"

df.drop('employee_id', axis=1, inplace=True)


# Handling outliers

# for 'no_of_trainings':


df['no_of_trainings'] = winsorize(df['no_of_trainings'], limits=[0.05, 0.05])



# for 'age':

# Calculating Quartiles

Q1 = np.percentile(df['age'],25)
Q3 = np.percentile(df['age'],75)

# Calculating Inter Quartile Range

IQR = Q3 - Q1

# Calculating lower and upper limit

low_lim = Q1 - 1.5 * IQR
upp_lim = Q3 + 1.5 * IQR

# Cliping outilers

df['age'] = df['age'].clip(lower=low_lim,upper=upp_lim)


# for 'previous_year_rating':

# Calculating Quartiles

Q1 = np.percentile(df['previous_year_rating'],25)
Q3 = np.percentile(df['previous_year_rating'],75)

# Calculating Inter Quartile Range

IQR = Q3 - Q1

# Calculating lower and upper limit

low_lim = Q1 - 1.5 * IQR
upp_lim = Q3 + 1.5 * IQR

# Cliping outilers

df['previous_year_rating'] = df['previous_year_rating'].clip(lower=low_lim,upper=upp_lim)


# for 'length_of_service':

# Calculating Quartiles

Q1 = np.percentile(df['length_of_service'],25)
Q3 = np.percentile(df['length_of_service'],75)

# Calculating Inter Quartile Range

IQR = Q3 - Q1

# Calculating lower and upper limit

low_lim = Q1 - 1.5 * IQR
upp_lim = Q3 + 1.5 * IQR

# Cliping outilers

df['length_of_service'] = df['length_of_service'].clip(lower=low_lim,upper=upp_lim)













