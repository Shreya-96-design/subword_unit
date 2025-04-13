Subword Unit Duration Modeling in Speech

This project performs analysis on phoneme-level subword features extracted from English speech data. 
It classifies speech as either native or non-native and also classifies the phoneme type (consonant/vowel) using features like phoneme duration, class, and identity.
Folder structure 
subword_unit/
├── data/
│   ├── american_english/       # Native speaker JSON files
│   └── other_english/          # Non-native speaker JSON files
├── phoneme_data.csv            # Generated dataset after extraction
├── sw.py                       # Main script 
└── README.md                   # You're reading it :)

How to run:-
Clone the repository 
-git clone https://github.com/Shreya-96-design/subword_unit.git
-cd subword_unit
Prepare dataset:-
subword_unit/
└── data/
    ├── american_english/
    │   └── *.json
    └── other_english/
        └── *.json
Install Libraries:-
pandas,sklearn, seaborn 
(eg.,-pip install pandas scikit-learn matplotlib seaborn)
Run script-
python sw.py
What you’ll see-
CSV file: phoneme_data.csv (extracted dataset)
Visualizations:
Phoneme durations by class and speaker type
Boxplots comparing native vs non-native durations
Feature importance bar plot
Accuracy & F1 comparisons for classifiers
Model training and evaluation:
-Random Forest
-Logistic Regression
-Decision Tree
Classification Reports and Confusion Matrices

Model Inputs
Features:
-subword_encoded
-class_encoded
-duration

Targets:
-speaker_type (native or non_native)
-class (vowel or consonant)


Models are evaluated using:

-Accuracy
-F1 Score
-Confusion Matrix
-Stratified K-Fold Cross-validation




