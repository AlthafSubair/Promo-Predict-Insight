import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Load dataset
df = pd.read_csv('train_LZdllcl.csv')
pd.set_option('display.max_columns', None)

# Handling null values
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
le_department = LabelEncoder()
df['department'] = le_department.fit_transform(df['department'])

df['region'] = df['region'].str.extract(r'(\d+)').astype(int)

od_education = OrdinalEncoder(categories=[["Below Secondary", "Bachelor's", "Master's & above"]], dtype=int)
df['education'] = od_education.fit_transform(df[['education']])

le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

od_recruitment_channel = OrdinalEncoder(categories=[["other", "sourcing", "referred"]], dtype=int)
df['recruitment_channel'] = od_recruitment_channel.fit_transform(df[['recruitment_channel']])

# Normalization
standard_scaler = StandardScaler()
standard_columns = ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score']
df[standard_columns] = standard_scaler.fit_transform(df[standard_columns])

# Separate the classes
cls_false = df[df['is_promoted'] == 0]
cls_true = df[df['is_promoted'] == 1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('is_promoted', axis=1), df['is_promoted'], test_size=0.2, random_state=119)

# Oversample the minority class only in the training set
oversample = resample(cls_true, n_samples=len(cls_false), random_state=119)
X_train_oversampled = pd.concat([X_train, oversample.drop('is_promoted', axis=1)])
y_train_oversampled = pd.concat([y_train, oversample['is_promoted']])

#function for passing features entered by uder
def ml_pred(user_input_df):
    
    # Apply the encoding from the original DataFrame to the user input
    
    user_input_df['department'] = le_department.transform(user_input_df['department'])
    user_input_df['region'] = user_input_df['region'].str.extract(r'(\d+)').astype(int)  # Assuming it's an integer
    user_input_df['education'] = od_education.transform(user_input_df[['education']])
    user_input_df['gender'] = le_gender.transform(user_input_df['gender'])
    user_input_df['recruitment_channel'] = od_recruitment_channel.transform(user_input_df[['recruitment_channel']])

    # Apply the scaling to user input features
    
    user_input_df[standard_columns] = standard_scaler.transform(user_input_df[standard_columns])


    # Initialize the Random Forest model
    
    model_RF = RandomForestClassifier(random_state=119)

    # Train the model with the oversampled training set
    
    model_RF.fit(X_train_oversampled, y_train_oversampled)


    # Make predictions on the user input
    
    user_pred = model_RF.predict(user_input_df)

    # Decoding result to human understandable form
    if user_pred == 1:
        user_pred = 'Employee is eligible for Promotion'
    else:
        user_pred = 'Employee is Not eligible for promotion'
    return user_pred