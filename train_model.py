import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv(r"C:\Users\TANUJA\Downloads\archive\amazon_delivery.csv")

# Drop rows with missing values (optional but safe)
df.dropna(inplace=True)

# Encode categorical features
label_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Select input features and target
X = df[['Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude',
        'Drop_Latitude', 'Drop_Longitude', 'Weather', 'Traffic',
        'Vehicle', 'Area', 'Category']]

y = df['Delivery_Time']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("delivery_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as delivery_model.pkl")
