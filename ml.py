import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the preprocessed data
df = pd.read_csv('preprocessed_data.csv')
pd.set_option('display.max_columns', None)

# Assuming 'data' is your DataFrame with features and target
X = df.drop('is_promoted', axis=1)
y = df['is_promoted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=119)

# Train the model
model_RF = RandomForestClassifier(random_state=119)
model_RF.fit(X_train, y_train)

# Take user input for features
user_input_features = []
for column in X.columns:
    value = input(f"Enter the value for {column}: ")
    user_input_features.append(value)

# Create a DataFrame with user input
user_input_df = pd.DataFrame([user_input_features], columns=X.columns)

# Make prediction on user input
user_prediction = model_RF.predict(user_input_df)

# Print the prediction

print(f"The predicted result for the user input is: {user_prediction}")


