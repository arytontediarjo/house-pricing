import pandas as pd
import numpy as np
from feature_engine import categorical_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import sklearn.metrics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

def combine_temporal(X_train, X_test):
    X_train["n_temp_houseAge"] = \
    X_train["YrSold"] - X_train["YearBuilt"]
    X_train["n_temp_lastRemodelled"] = \
    X_train["YrSold"] - X_train["YearRemodAdd"]
    X_train["n_temp_garageAge"] = \
    X_train["YrSold"] - X_train["GarageYrBlt"]
    
    X_test["n_temp_houseAge"] = \
    X_test["YrSold"] - X_test["YearBuilt"]
    X_test["n_temp_lastRemodelled"] = \
    X_test["YrSold"] - X_test["YearRemodAdd"]
    X_test["n_temp_garageAge"] = \
    X_test["YrSold"] - X_test["GarageYrBlt"]
    
    cols = ["YrSold",
           "YearBuilt",
           "YearRemodAdd",
            "GarageYrBlt", 
           "MoSold"]
    
    return X_train.drop(cols, axis = 1), X_test.drop(cols, axis = 1)

def fill_with_median(X_train, X_test, variable):
    median = X_train[variable].median()
    X_train[variable] = X_train[variable].fillna(median)
    X_test[variable] = X_test[variable].fillna(median)
    return X_train, X_test

def fill_zero(X_train, X_test, variable):
    X_train[variable] = X_train[variable].fillna(0)
    X_test[variable] = X_test[variable].fillna(0)
    return X_train, X_test

def fill_with_missing(X_train, X_test, variable):
    X_train[variable] = X_train[variable].fillna("Missing")
    X_test[variable] = X_test[variable].fillna("Missing")
    return X_train, X_test

def fill_with_mode_from_train(X_train, X_test, variable):
    mode = X_train[variable].mode().iloc[0]
    X_train[variable] = X_train[variable].fillna(mode)
    X_test[variable] = X_test[variable].fillna(mode)
    return X_train, X_test

def find_non_rare_labels(df, variable, tolerance):
    temp = df.groupby([variable])[variable].count()/len(df)
    non_rare = [x for x in temp.loc[temp > tolerance].index.values]
    return non_rare

def rare_encoding(X_train, X_test, variable, tolerance, n_cat):
    encoder = ce.RareLabelCategoricalEncoder(tol = tolerance, 
                                             n_categories= n_cat,
                                             variables=variable,
                                             replace_with='Rare', 
                                             return_object=True)
    # fit the encoder
    encoder.fit(X_train)
    return encoder.transform(X_train), encoder.transform(X_test)

def one_hot_encoding(X_train, X_test, variable):
    encoder = ce.OneHotCategoricalEncoder(
        top_categories = None,
        variables = variable,
        drop_last = True)
    encoder.fit(X_train)
    return encoder.transform(X_train), encoder.transform(X_test)

def label_encoding(X_train, 
                   y_train,
                   X_test, 
                   variable, 
                   encoding_method):
    encoder = ce.OrdinalCategoricalEncoder(
        encoding_method = encoding_method,
        variables = variable)
    
    if encoding_method == "ordered":
        encoder.fit(X_train, y_train)
    else:
        encoder.fit(X_train)
    return encoder.transform(X_train), encoder.transform(X_test)

def power_transformer(X_train, X_test, variable):
    tf = vt.PowerTransformer(variables = variable, exp=0.5)
    tf.fit(X_train)
    return tf.transform(X_train), tf.transform(X_test)


def boxcox_transformer(X_train, X_test, variable):
    for i in variable:
        X_train[i] = boxcox1p(X_train[i],
                              boxcox_normmax(X_train[i] + 1))
        X_test[i] = boxcox1p(X_test[i], 
                             boxcox_normmax(X_train[i] + 1))
    return X_train, X_test


def log_transformer(data):
    return np.log(data)

def assess_performance(X_train, 
                        X_test, 
                        y_train, 
                        y_test):
        
    lr = LinearRegression().fit(X_train, y_train)
    lasso = linear_model.Lasso(alpha = 0.001).fit(X_train, y_train)
    ridge = linear_model.Ridge(alpha = 1).fit(X_train,y_train)
    rf = RandomForestRegressor(
        random_state = 100, n_estimators = 100, max_depth = 10).fit(X_train, y_train)
    gb = GradientBoostingRegressor(
        random_state= 100, n_estimators = 50, max_depth = 5).fit(X_train, y_train)
    
    cross_val = {}
    cross_val["model"] = []
    cross_val["rmse"] = []
    test_set = {} 
    for model, name in zip([lasso, 
                            ridge, 
                            gb, 
                            rf], 
                            ["lasso", 
                            "ridge", 
                            "gradientboost", 
                            "randomforest"]):
       
        test_set[name + "_test_rmse"] = \
            sklearn.metrics.mean_squared_error(model.predict(X_test), y_test, squared = False)
        cross_val["model"].append("{}".format(name))
        cross_val["rmse"].append(
            np.sqrt(-1 * cross_val_score(model, X_train, y_train, 
                                         scoring = "neg_mean_squared_error", cv = 8)))
    return cross_val, test_set