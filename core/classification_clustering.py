import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def get_feature_names(df):
    return df.columns.tolist()
def run_classification(df, target_column, algorithm, params):
    X = df.drop(columns=[target_column]).select_dtypes(exclude=['object']) # <--- زود دي هنا
    y = df[target_column]
    if y.dtype == 'float64' or y.dtype == 'float32':
        y = y.astype('int') 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    if algorithm == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=params.get("max_depth"))
    elif algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=params.get("n_estimators", 100))
    elif algorithm == "KNN":
        model = KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))
    elif algorithm == "SVM":
        model = SVC(C=params.get("C", 1.0))
    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return model, accuracy, report, X_test, y_test
def run_clustering(df, n_clusters):
    numeric_df = df.select_dtypes(include=['number'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(numeric_df)
    return kmeans, clusters

def run_bayesian(df, target_column):
    from sklearn.naive_bayes import GaussianNB
    X = df.drop(columns=[target_column]).select_dtypes(exclude=['object'])
    y = df[target_column]
    if y.dtype in ['float64', 'float32']:
        y = y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return model, accuracy, report, X_test, y_test

def run_neural_network(df, target_column, hidden_layer_sizes=(100,), max_iter=300):
    from sklearn.neural_network import MLPClassifier
    X = df.drop(columns=[target_column]).select_dtypes(exclude=['object'])
    y = df[target_column]
    if y.dtype in ['float64', 'float32']:
        y = y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return model, accuracy, report, X_test, y_test
