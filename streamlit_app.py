import streamlit as st
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Laden der CSV-Datei
df = pd.read_csv("email.csv")

# Daten vorbereiten
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
X = df['Message']
Y = df['spam']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Vorverarbeitung der Trainingsdaten
def tokenize(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words and token.isalpha()]
    return tokens

X_train_processed = [' '.join(tokenize(text)) for text in X_train]

# Modelle definieren
def create_pipeline(model):
    return Pipeline([
        ('vectorizer', CountVectorizer(analyzer='word', lowercase=True)),
        ('tfidf', TfidfTransformer()),
        ('classifier', model)
    ])

# Modelle trainieren
clf_NaiveBayes = create_pipeline(MultinomialNB())
clf_NaiveBayes.fit(X_train_processed, y_train)

clf_svm = create_pipeline(SVC(kernel="rbf", C=1000, gamma=0.001))
clf_svm.fit(X_train_processed, y_train)

clf_knn = create_pipeline(KNeighborsClassifier(n_neighbors=3))
clf_knn.fit(X_train_processed, y_train)

clf_DecisionTree = create_pipeline(DecisionTreeClassifier())
clf_DecisionTree.fit(X_train_processed, y_train)

clf_rf = create_pipeline(RandomForestClassifier(n_estimators=100))
clf_rf.fit(X_train_processed, y_train)

# Streamlit App
st.title("Email Spam Detector")
st.write("Enter an email to see if it's spam or ham.")

user_input = st.text_area("Email Content", "")

model_option = st.selectbox(
    "Choose a model for prediction:",
    ("Naive Bayes", "SVM", "KNN", "Decision Tree", "Random Forest")
)

if st.button("Predict"):
    if user_input:
        input_processed = ' '.join(tokenize(user_input))
        if model_option == "Naive Bayes":
            prediction = clf_NaiveBayes.predict([input_processed])
        elif model_option == "SVM":
            prediction = clf_svm.predict([input_processed])
        elif model_option == "KNN":
            prediction = clf_knn.predict([input_processed])
        elif model_option == "Decision Tree":
            prediction = clf_DecisionTree.predict([input_processed])
        elif model_option == "Random Forest":
            prediction = clf_rf.predict([input_processed])

        if prediction == 1:
            st.write("The model predicts: **Spam**")
        else:
            st.write("The model predicts: **Ham**")
    else:
        st.write("Please enter text to get a prediction.")

# Berechnung der Metriken
def calculate_metrics(model, X_test, y_test):
    X_test_processed = [' '.join(tokenize(text)) for text in X_test]
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

naive_metrics = calculate_metrics(clf_NaiveBayes, X_test, y_test)
svm_metrics = calculate_metrics(clf_svm, X_test, y_test)
knn_metrics = calculate_metrics(clf_knn, X_test, y_test)
dt_metrics = calculate_metrics(clf_DecisionTree, X_test, y_test)
rf_metrics = calculate_metrics(clf_rf, X_test, y_test)

# Anzeige der Metriken
st.write("### Model Metrics")

metrics_df = pd.DataFrame({
    "Model": ["Naive Bayes", "SVM", "KNN", "Decision Tree", "Random Forest"],
    "Accuracy": [naive_metrics[0], svm_metrics[0], knn_metrics[0], dt_metrics[0], rf_metrics[0]],
    "Precision": [naive_metrics[1], svm_metrics[1], knn_metrics[1], dt_metrics[1], rf_metrics[1]],
    "Recall": [naive_metrics[2], svm_metrics[2], knn_metrics[2], dt_metrics[2], rf_metrics[2]],
    "F1-Score": [naive_metrics[3], svm_metrics[3], knn_metrics[3], dt_metrics[3], rf_metrics[3]]
})

st.write(metrics_df)

# Visualisierung der Metriken
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

sns.barplot(x='Model', y='Accuracy', data=metrics_df, ax=ax[0, 0])
ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_title('Accuracy')

sns.barplot(x='Model', y='Precision', data=metrics_df, ax=ax[0, 1])
ax[0, 1].set_ylim(0, 1)
ax[0, 1].set_title('Precision')

sns.barplot(x='Model', y='Recall', data=metrics_df, ax=ax[1, 0])
ax[1, 0].set_ylim(0, 1)
ax[1, 0].set_title('Recall')

sns.barplot(x='Model', y='F1-Score', data=metrics_df, ax=ax[1, 1])
ax[1, 1].set_ylim(0, 1)
ax[1, 1].set_title('F1-Score')

st.pyplot(fig)

#Sidebar
st.sidebar.header("Check Your Emails for Spam")
st.sidebar.write("This app uses advanced text processing and machine learning techniques to ensure reliable spam detection.")
st.sidebar.markdown("#### Motivation")
st.sidebar.markdown("- Emails are a crucial means of communication")
st.sidebar.markdown("- Spam and phishing attacks are becoming more sophisticated")
st.sidebar.markdown("- Email providers want to offer maximum security to their customers")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("## Contributors")
st.sidebar.markdown("Jonathan Bauer, Aryan Rajput, Leon Hirschpeck, Behnam Irani")
