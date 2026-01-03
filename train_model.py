import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/churn.csv")

# Convert target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(0, inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    if col != 'customerID':
        df[col] = le.fit_transform(df[col])

# Features & target
# Select only required features
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# SAVE MODEL & SCALER
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
