import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("heart.csv")
print(data.head())
corrmat=data.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(16,16))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap='RdYlGn')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



standardScaler = StandardScaler()
columns_to_scale = ['age', 'resting bp s', 'cholesterol','max heart rate', 'oldpeak']



y = data['target']
X = data.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)



age=int(st.sidebar.text_input("Enter the Age"))
age=float(age)
sex=st.sidebar.radio("Enter the Sex",["Male","Female"])
if sex=="Male":
    sex=1
else:
    sex=0
oldpeak=float(st.sidebar.text_input("Oldpeak",0))
oldpeak=float(oldpeak)
Sk=float(st.sidebar.text_input("Serum Cholestrol",0))
RBPS=float(st.sidebar.text_input("Enter the resting blood pressure"))
CPT=st.sidebar.radio("Enter Chest Pain Type",["typical angina","atypical angina","non-anginal pain","asymptomatic"])
if CPT=="typical angina":
    CPT=1
elif CPT=="atypical angina":
    CPT=2
elif CPT=="non-anginal pain":
    CPT=3
else:
    CPT=4

MHR=float(st.sidebar.slider("MAX Heart Rate",min_value=71,max_value=202))

FBS=st.sidebar.radio("Blood Sugar level above or below",["Above 120mg","Below or Equal"])
if FBS=="Above 120mg":
    FBS=1
else:
    FBS=0

RECG=st.sidebar.radio("Resting ECG",["Normal","ST wave abnormality","Showing Probable"])
if RECG=="Normal":
    RECG=0
elif RECG=="ST wave abnormality":
    RECG=1
else:
    RECG=2
EIA=st.sidebar.radio("Exercise induced angina",["YES","NO"])
if EIA=="YES":
    EIA=1
else:
    EIA=0

SLOPE=st.sidebar.radio("Slope of the peak exercise ST segment",["Unsloping","Flat","Downsloping"])
if SLOPE=="Unsloping":
    SLOPE=1
elif SLOPE=="Flat":
    SLOPE=2
else:
    SLOPE=3


z=np.array([[age,sex,CPT,RBPS,Sk,FBS,RECG,MHR,EIA,oldpeak,SLOPE]])

import numpy as np

def prea(a):
    if a[0]==1:
        return "Heart disease"
    else:
        return "Normal"



fn=data[["age","sex","chest pain type","resting bp s","cholesterol","fasting blood sugar","resting ecg","max heart rate","exercise angina","oldpeak","ST slope"]]
cn=data[["target"]]
from sklearn.tree import plot_tree 
def LR():
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression()
    model.fit(X_train,y_train)
    score=model.score(X_test,y_test)
    st.write(score*100)
    a=model.predict(z)
    pre=prea(a)
    st.write(pre)

def DT():
    from sklearn.tree import plot_tree
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.write(f"Accuracy: {score * 100:.2f}%")

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(20, 10))  # Increase size if needed
    plot_tree(model, feature_names=X.columns, class_names=['Normal', 'Heart disease'], filled=True, ax=ax)
    st.pyplot(fig)  # Make sure `fig` is passed correctly

    # Make prediction
    a = model.predict(z)
    pre = prea(a)
    st.write(pre)
def SVM():
    from sklearn.svm import SVC
    model=SVC()
    model.fit(X_train,y_train)
    score=model.score(X_test,y_test)
    st.write(score*100)
    a=model.predict(z)
    pre=prea(a)
    st.write(pre)

def RFC():
    a=st.sidebar.text_input("Enter n estimators")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import plot_tree
    model=RandomForestClassifier(n_estimators=int(a))
    model.fit(X_train,y_train)
    score=model.score(X_test,y_test)
    st.write(score*100)
    a=model.predict(z)
    pre=prea(a)
    st.write(pre)
    num_trees = min(len(model.estimators_), 5)  # Plot at most 5 trees
    for index in range(num_trees):
        fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the size if needed
        plot_tree(model.estimators_[index], feature_names=X.columns, class_names=['Normal', 'Heart disease'], filled=True, ax=ax)
        st.pyplot(fig)

def XGB():
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.write(f"Accuracy: {score * 100:.2f}%")
    import xgboost as xgb
    num_trees = min(len(model.get_booster().get_dump()), 5)  # Plot at most 5 trees
    for index in range(num_trees):
        fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the size if needed
        xgb.plot_tree(model, num_trees=index, ax=ax)
        st.pyplot(fig)
    # Make prediction
    a = model.predict(z)
    pre = prea(a)
    st.write(pre)

    # Plot a few trees
    

def GBC(): 
    from sklearn.ensemble import GradientBoostingClassifier
    model=GradientBoostingClassifier()
    model.fit(X_train,y_train)
    score=model.score(X_test,y_test)
    st.write(score*100)
    a=model.predict(z)
    pre=prea(a)
    st.write(pre)

def KNN():
    from sklearn.neighbors import KNeighborsClassifier
    model=KNeighborsClassifier()
    model.fit(X_train,y_train)
    score=model.score(X_test,y_test)
    st.write(score*100)
    a=model.predict(z)
    pre=prea(a)
    st.write(pre)

def MLP():
    from sklearn.neural_network import MLPClassifier
    model=MLPClassifier()
    model.fit(X_train,y_train)
    score=model.score(X_test,y_test)
    st.write(score*100)
    a=model.predict(z)
    pre=prea(a)
    st.write(pre)





sel=st.sidebar.selectbox("Enter the algorithm",("Logistic Regression","Decision Tree","Support Vector Classifier",
"Random Forest Classifier","XGBRegression","Gradient Boosting Classifier","MLP Classifier"))

if sel=="Logistic Regression":
    LR()
elif sel=="Decision Tree":
    DT()
elif sel=="Support Vector Classifier":
    SVM()
elif sel=="Random Forest Classifier":
    RFC()
elif sel=="Gradient Boosting Classifier":
    GBC()
elif sel=="K-Nearest Neighbours":
    KNN()
else:
    MLP()


