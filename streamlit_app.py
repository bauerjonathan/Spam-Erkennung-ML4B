# streamlit_app.py
import streamlit as st
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Laden der CSV-Datei
df = pd.read_csv("emai.csv")

# Daten vorbereiten
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
X = df['Message']
Y = df['spam']
X_train, X_test, y_train, y_test = train_test_split(X, Y)

# Modelle definieren
def create_pipeline(model):
    return Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', model)
    ])

# Modelle trainieren
clf_NaiveBaised = create_pipeline(MultinomialNB())
clf_NaiveBaised.fit(X_train, y_train)

clf_svm = create_pipeline(SVC(kernel="rbf", C=1000, gamma=0.001))
clf_svm.fit(X_train, y_train)

clf_knn = create_pipeline(KNeighborsClassifier(n_neighbors=3))
clf_knn.fit(X_train, y_train)

clf_DecisionTree = create_pipeline(DecisionTreeClassifier())
clf_DecisionTree.fit(X_train, y_train)

clf_rf = create_pipeline(RandomForestClassifier(n_estimators=100))
clf_rf.fit(X_train, y_train)

# Streamlit App
st.title("Email Spam Detector")
st.write("Geben Sie eine E-Mail ein, um zu sehen, ob sie Spam oder Ham ist.")

user_input = st.text_area("E-Mail Inhalt", "")

model_option = st.selectbox(
    "Wählen Sie ein Modell zur Vorhersage aus:",
    ("Naive Bayes", "SVM", "KNN", "Decision Tree", "Random Forest")
)

if st.button("Vorhersagen"):
    if user_input:
        if model_option == "Naive Bayes":
            prediction = clf_NaiveBaised.predict([user_input])
        elif model_option == "SVM":
            prediction = clf_svm.predict([user_input])
        elif model_option == "KNN":
            prediction = clf_knn.predict([user_input])
        elif model_option == "Decision Tree":
            prediction = clf_DecisionTree.predict([user_input])
        elif model_option == "Random Forest":
            prediction = clf_rf.predict([user_input])

        if prediction == 1:
            st.write("Das Modell sagt: **Spam**")
        else:
            st.write("Das Modell sagt: **Ham**")
    else:
        st.write("Bitte geben Sie einen Text ein, um eine Vorhersage zu erhalten.")

st.write("### Modell-Genauigkeiten")
naive_acc = accuracy_score(y_test, clf_NaiveBaised.predict(X_test))
svm_acc = accuracy_score(y_test, clf_svm.predict(X_test))
knn_acc = accuracy_score(y_test, clf_knn.predict(X_test))
dt_acc = accuracy_score(y_test, clf_DecisionTree.predict(X_test))
rf_acc = accuracy_score(y_test, clf_rf.predict(X_test))

accuracies = {
    "Naive Bayes": naive_acc,
    "SVM": svm_acc,
    "KNN": knn_acc,
    "Decision Tree": dt_acc,
    "Random Forest": rf_acc
}

st.bar_chart(pd.Series(accuracies) * 100)
