
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
import pickle

def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['has_double_slash'] = 1 if '//' in url else 0
    features['has_dash'] = 1 if '-' in url else 0
    features['has_percent'] = 1 if '%' in url else 0
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_at'] = url.count('@')
    features['num_slash'] = url.count('/')
    features['num_question'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_ampersand'] = url.count('&')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_letters'] = sum(c.isalpha() for c in url)
    return features

# Load the dataset
df = pd.read_csv('advanced_link_security_scanner/dataset/PhiUSIIL_Phishing_URL_Dataset.csv')

# Drop rows with missing values in 'URL' or 'label' columns
df.dropna(subset=['URL', 'label'], inplace=True)

# Convert labels to numerical (0 for benign, 1 for phishing)
df['label'] = df['label'].apply(lambda x: 1 if x == 'phishing' else 0)

# Extract features
features_df = df['URL'].apply(lambda x: pd.Series(extract_features(x)))

X = features_df
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Save the model and feature names
with open('advanced_link_security_scanner/url_classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('advanced_link_security_scanner/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print('Model and feature names saved successfully.')


