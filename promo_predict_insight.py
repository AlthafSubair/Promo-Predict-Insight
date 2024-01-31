import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('train_LZdllcl.csv')
pd.set_option('display.max_columns', None)

# Handling missing values
df['education'] = df['education'].fillna(df['education'].mode()[0])
df['previous_year_rating'] = df['previous_year_rating'].fillna(df['previous_year_rating'].median())

# Drop 'employee_id'
df.drop('employee_id', axis=1, inplace=True)


# Handling outliers
columns_to_clip = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service']

for column in columns_to_clip:
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1
    low_lim = Q1 - 1.5 * IQR
    upp_lim = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=low_lim, upper=upp_lim)
    
    # Encoding
le = LabelEncoder()
df['department'] = le.fit_transform(df['department'])

df['region'] = df['region'].str.extract('(\d+)').astype(int)

od_education = OrdinalEncoder(categories=[["Below Secondary", "Bachelor's", "Master's & above"]], dtype=int)
df['education'] = od_education.fit_transform(df[['education']])

df['gender'] = le.fit_transform(df['gender'])

od_recruitment_channel = OrdinalEncoder(categories=[["other", "sourcing", "referred"]], dtype=int)
df['recruitment_channel'] = od_recruitment_channel.fit_transform(df[['recruitment_channel']])

# Normalization
standard_scaler = StandardScaler()
standard_columns = ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score']
df[standard_columns] = standard_scaler.fit_transform(df[standard_columns])

# Assuming 'data' is your DataFrame with features and target
X = df.drop('is_promoted', axis=1)
y = df['is_promoted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119)

# Train model
model_RF = RandomForestClassifier(random_state=119)
model_RF.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_RF.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')