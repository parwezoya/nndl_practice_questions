import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset 
# You can download 'train.csv' from Kaggle
df = pd.read_csv('train.csv')

# 2. DATA CLEANING (The most important part!)
# We select the features that actually matter
# Pclass = Ticket Class, Sex = Gender, Age, SibSp = Siblings, Parch = Parents, Fare
data = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
target = df['Survived']

# Handle Missing Values: Fill missing ages with the average age
data['Age'] = data['Age'].fillna(data['Age'].mean())

# Convert Text to Numbers: Male=0, Female=1
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# 3. SPLITTING & SCALING
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Scaling: This ensures the NN doesn't get confused by large numbers (Fare)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. BUILDING THE NEURAL NETWORK
model = models.Sequential([
    # Input layer: 6 features -> 16 neurons
    layers.Dense(16, activation='relu', input_shape=(6,)), 
    layers.Dropout(0.2), # Prevent overfitting
    
    # Hidden layer: 8 neurons
    layers.Dense(8, activation='relu'),
    
    # Output layer: 1 neuron (Survived or Not)
    layers.Dense(1, activation='sigmoid') 
])

# 5. COMPILING & TRAINING
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training on Titanic Data...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# 6. EVALUATION
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nAccuracy on Test Data: {accuracy*100:.2f}%")

# --- TEST WITH A CUSTOM PASSENGER ---
# [Pclass=3, Sex=Female(1), Age=22, SibSp=1, Parch=0, Fare=7.25]
passenger = np.array([[3, 1, 22, 1, 0, 7.25]])
passenger_scaled = scaler.transform(passenger)
prediction = model.predict(passenger_scaled)

print(f"Survival Probability: {prediction[0][0]*100:.2f}%")
print("Result:", "Survived" if prediction[0][0] > 0.5 else "Did not survive")