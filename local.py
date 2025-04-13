import json
import os
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


def assess_json_files(folders):
    for folder in folders:
        print(f"Assessing folder: {folder}")
        for root, _, files in os.walk(folder):
          #print(files)
    
          for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            #print(data)

                        #print(f"Valid JSON file: {file_path}")
                        

                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON file: {file_path} - Error: {e}")
                    except Exception as e:
                        print(f"Error reading file: {file_path} - Error: {e}")
                    

# Example usage
folders_to_assess = [
    '/Users/shreyagupta/Downloads/american_english/',
    '/Users/shreyagupta/Downloads/other_english/'
]

assess_json_files(folders_to_assess)


def extract_subword_data(folders):
    records = []  # to store extracted data
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)

                        #print(f"Extracting from: {file_path}")

                        #print("Top-level keys in JSON:", json_data.keys())
                        #print("Sample content:", json.dumps(json_data, indent=2)[:1000])  # print first 1000 chars to not overload
                            
                        # to check 'result' and 'words' keys exist in this path
                        if 'result' not in json_data:
                            #print(f"No 'result' key found in JSON for file: {file_path}")
                            continue
                        
                        #print("All keys under 'result':", json_data['result'].keys())
                        

                        if 'segments' not in json_data['result']:
                            #print(f"No 'segments' key found in 'result' for file: {file_path}")
                            continue

                
                        segments = json_data['result']['segments']
                        for segment in segments:
                            words = segment.get('words', [])
                            for word in words:
                                if not word.get('oov', False):  # Skip OOV words
                                    for phone in word.get('phones', []):
                                        if phone.get('class', '').lower() != 'sil':  # Skip silence
                                            records.append({
                                                'subword': phone.get('phone', ''),
                                                'duration': phone.get('duration', 0.0),
                                                'class': phone.get('class', ''),
                                                'speaker_type': 'native' if 'american' in folder else 'non_native'
                                            })
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
    return pd.DataFrame(records)


# Extract data into a DataFrame
df = extract_subword_data(folders_to_assess)

# Display the first few rows of the DataFrame
print("DF shape:", df.shape)
print("DF columns:", df.columns)
print("DF head:\n", df.head())


from sklearn.preprocessing import LabelEncoder

# Encode subwords and phoneme classes
subword_encoder = LabelEncoder()
class_encoder = LabelEncoder()


if df.empty or 'subword' not in df.columns:
    print("No data extracted. Please check your JSON structure or extraction logic.")
    exit()


df['subword_encoded'] = subword_encoder.fit_transform(df['subword'])
df['class_encoded'] = class_encoder.fit_transform(df['class'])

# Display the updated DataFrame
print(df.head())

sns.boxplot(data=df, x='class', y='duration', hue='speaker_type')
plt.title("Phoneme Duration by Class and Speaker Type")
plt.show()

df.to_csv("phoneme_data.csv", index=False)
print(df['speaker_type'].value_counts())
print(df['class'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Features and label
X = df[['subword_encoded', 'class_encoded', 'duration']]
y = df['speaker_type']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


feature_names = X.columns
importances = clf.feature_importances_

plt.figure(figsize=(6, 4))
plt.barh(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# Same features, different label
y_class = df['class']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
clf_c = RandomForestClassifier(n_estimators=100, random_state=42)
clf_c.fit(X_train_c, y_train_c)

y_pred_c = clf_c.predict(X_test_c)

print("\nPhoneme Class (C/V) Classification Report:\n", classification_report(y_test_c, y_pred_c))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_c, y_pred_c))

# Avg duration per subword
plt.figure(figsize=(12, 5))
sns.barplot(data=df, x='class', y='duration')
plt.title("Average Duration by Phoneme Class")
plt.show()

# Distribution of durations for native vs non-native
plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x='speaker_type', y='duration')
plt.title("Duration Distribution by Speaker Type")
plt.show()


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# Encode target labels
y_encoded = LabelEncoder().fit_transform(y)

# Defining models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Decision Tree": DecisionTreeClassifier(),
}

# Stratified k-fold cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluation
results = {}
for name, model in models.items():
    accuracy = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy').mean()
    f1 = cross_val_score(model, X, y_encoded, cv=cv, scoring=make_scorer(f1_score)).mean()
    results[name] = {'Accuracy': accuracy, 'F1 Score': f1}
    print(f"{name} -> Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Separate accuracy and F1 scores
model_names = list(results.keys())
accuracies = [results[model]['Accuracy'] for model in model_names]
f1_scores = [results[model]['F1 Score'] for model in model_names]

x = range(len(model_names))

plt.figure(figsize=(10, 5))

# Accuracy bars
plt.bar(x, accuracies, width=0.4, label='Accuracy', align='center', color='skyblue')
# F1 Score bars 
plt.bar([i + 0.4 for i in x], f1_scores, width=0.4, label='F1 Score', align='center', color='salmon')

# Labels and styling
plt.xticks([i + 0.2 for i in x], model_names)
plt.ylabel('Score')
plt.title('Model Performance: Accuracy vs F1 Score')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


                     
                                   


