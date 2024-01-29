import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn.ensemble import RandomForestClassifier

# Load dataset

df = pd.read_csv('preprocessed_data.csv')
pd.set_option('display.max_columns', None)

# Assuming 'data' is your DataFrame with features and target
X = df.drop('is_promoted', axis=1)
y = df['is_promoted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119)


model_RF = RandomForestClassifier(random_state=119)
model_RF.fit(X_train,y_train)

# Make predictions on the testing data
y_pred = model_RF.predict(X_test)

# Print classification report for more detailed evaluation
print(classification_report(y_test, y_pred))


