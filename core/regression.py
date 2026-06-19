import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def get_model(model_name, params):
    if model_name == "LinearRegression":
        return LinearRegression() # الـ Linear هنا مش محتاج params
    elif model_name == "Ridge":
        return Ridge(**params)      # <-- ضيف النجمتين دول هنا
    elif model_name == "Lasso":
        return Lasso(**params)      # <-- وضيقهم هنا كمان للأمان
    elif model_name == "DecisionTree":
        return DecisionTreeRegressor(**params) # <-- وهنا
    elif model_name == "RandomForest":
        return RandomForestRegressor(**params) # <-- وهنا
    else:
        raise ValueError("Invalid model name")
    

def regression_model(X, y, model_name, params={}, preprocessor=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    if preprocessor:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
    model = get_model(model_name, params)
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    Test = r2_score(y_test, y_pred)
    results = {
        "mean": scores.mean(),
        "Test": Test,
        "MAE": mae }  
    return model, y_test, y_pred, results