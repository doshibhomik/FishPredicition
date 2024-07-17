import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
file_path = 'Fish.csv'
fish_data = pd.read_csv(file_path)

# Encode the 'Species' column
label_encoder = LabelEncoder()
fish_data['Species'] = label_encoder.fit_transform(fish_data['Species'])

# Split the data into features and target variable
X = fish_data.drop(['Species', 'Weight'], axis=1)
y = fish_data['Species']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model, scaler, and label encoder
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
