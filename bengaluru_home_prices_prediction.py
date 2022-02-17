import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Bengaluru_House_Data.csv')

df = df.drop(['area_type', 'availability', 'balcony', 'society'], axis=1)

df.isnull().sum()

df = df.dropna()

df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

df['price_per_sqft'] = df['price']*100000 / df['total_sqft']

df.location = df.location.apply(lambda x: x.strip())

location_stats = df.groupby('location')['location'].count()

location_stats_less_than_10 = location_stats[location_stats <= 10]

df.location = df.location.apply(
    lambda x: 'other' if x in location_stats_less_than_10 else x)

df = df[~(df.total_sqft/df.bhk < 300)]


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st))
                           & (subdf.price_per_sqft < (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df = remove_pps_outliers(df)


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]}
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if(stats and stats['count'] > 5):
                exclude_indices = np.append(
                    exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')


df = remove_bhk_outliers(df)

df = df[df.bath < df.bhk+2]

df = df.drop(['size', 'price_per_sqft'], axis=1)

dummies = pd.get_dummies(df.location)

df = pd.concat([df, dummies.drop('other', axis=1)], axis=1)

df = df.drop('location', axis=1)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
y_pred = lr_clf.predict(X_test)


# def find_best_model_using_gridsearch(X, y):
#     algos = {
#         'linear_regression': {
#             'model': LinearRegression(),
#             'params': {
#                 'normalize': [True, False]
#             }},
#         'lasso': {
#             'model': Lasso(),
#             'params': {
#                 'alpha': [1, 2],
#                 'selection': ['random', 'cyclic']
#             }
#         },
#         'decision_tree': {
#             'model': DecisionTreeRegressor(),
#             'params': {
#                 'criterion': ['mse', 'friedman_mse'],
#                 'splitter': ['best', 'random']
#             }
#         }
#     }

#     scores = []
#     cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#     for algo_name, config in algos.items():
#         gs = GridSearchCV(config['model'], config['params'],
#                           cv=cv, return_train_score=False)
#         gs.fit(X, y)
#         scores.append(
#             {'model': algo_name, 'best_score': gs.best_score_, 'best_params': gs.best_params_})

#     return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


# models = find_best_model_using_gridsearch(scaled_X_train, y_train)
# print(models.head())

print(lr_clf.score(X_test, y_test))


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]


predicted_price1 = predict_price('1st Phase JP Nagar', 1000, 2, 2)
predicted_price2 = predict_price('Whitefield', 1000, 3, 3)

print(predicted_price2)

dump(lr_clf, 'bengaluru_home_prices.joblib')
