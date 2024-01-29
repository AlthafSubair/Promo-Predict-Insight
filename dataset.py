import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,StandardScaler

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

# Calculating Quartiles

Q1 = np.percentile(df['no_of_trainings'],25)
Q3 = np.percentile(df['no_of_trainings'],75)

# Calculating Inter Quartile Range

IQR = Q3 - Q1

# Calculating lower and upper limit

low_lim = Q1 - 1.5 * IQR
upp_lim = Q3 + 1.5 * IQR

# Cliping outilers

df['no_of_trainings'] = df['no_of_trainings'].clip(lower=low_lim,upper=upp_lim)


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


# Encoding

# Performed label encoding in 'department':

le = LabelEncoder()
df['department'] = le.fit_transform(df[['department']])

# performed transformation in 'region':

df['region'] = df['region'].str.extract('(\d+)').astype(int) # Extracted numerical data from the text and used it 



# performed ordinal encoding in 'education':

od = OrdinalEncoder()
od = OrdinalEncoder(categories=[["Below Secondary","Bachelor's","Master's & above"]],dtype=int)
df['education'] = od.fit_transform(df[['education']])


# Performed label encoding in 'gender':

le = LabelEncoder()
df['gender'] = le.fit_transform(df[['gender']])

# performed ordinal encoding in 'recruitment_channel':

od = OrdinalEncoder()
od = OrdinalEncoder(categories=[["other","sourcing","referred"]],dtype=int)
df['recruitment_channel'] = od.fit_transform(df[['recruitment_channel']])


# Normilazation

# Initialize StandardScaler
standard_scaler = StandardScaler()

# List of columns to be Standard scaled
standard_columns = ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score']

# Apply Standard Scaling
df[standard_columns] = standard_scaler.fit_transform(df[standard_columns])


# Downloading csv file for perfoming ml algorthims

df.to_csv('preprocessed_data.csv', index=False)















