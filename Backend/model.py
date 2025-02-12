import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Define the path to the dataset
DATA_PATH = os.path.join('..','dataset', "animal_disease_dataset.csv")

# Load the data
data = pd.read_csv(DATA_PATH)

# Preprocessing steps
#Label Encoding Diseases
def label_encode_columns(data,columns):
    lb = LabelEncoder()
    for column in columns:
        data[column] = lb.fit_transform(data[column])
    return data

col_to_encode = ['Animal','Disease']

data = label_encode_columns(data, col_to_encode)

# Replace symptoms features with actual symptoms
new_features = ['blisters on gums', 'blisters on hooves', 'blisters on mouth',
                'blisters on tongue', 'chest discomfort', 'chills', 'crackling sound',
                'depression', 'difficulty walking', 'fatigue', 'lameness', 'loss of appetite',
                'painless lumps', 'shortness of breath', 'sores on gums', 'sores on hooves',
                'sores on mouth', 'sores on tongue', 'sweats', 'swelling in abdomen',
                'swelling in extremities', 'swelling in limb', 'swelling in muscle',
                'swelling in neck']
#create columns
for feature in new_features:
    data[feature] = 0

# Update the new columns
for index, row in data.iterrows():
    for symptom_column in ['Symptom 1', 'Symptom 2', 'Symptom 3']:
        symptom = row[symptom_column]
        if symptom in new_features:
            data.loc[index, symptom] = 1

#Remove redundant colums
data.drop(['Symptom 1','Symptom 2','Symptom 3'], axis=1, inplace=True)

#Make diseae the last column
cols = list(data.columns)
cols.remove('Disease')
cols.append('Disease')
data = data[cols]

#train test split
X = data.drop("Disease", axis=1)
Y = data['Disease']
X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=50, random_state=12)

# Train the model
RForest = RandomForestClassifier(n_estimators = 100)
RForest.fit(X_train, y_train)

# Save the model to a file
model_filename = 'model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(RForest, file)

print(f"Model saved to {model_filename}")
