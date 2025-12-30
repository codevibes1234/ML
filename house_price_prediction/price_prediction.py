import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import uniform,randint

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
y_train = df_train.pop("SalePrice")

fill_zero_cols = [
        'PoolArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 
        '2ndFlrSF', 'LowQualFinSF', '1stFlrSF', '3SsnPorch', 
        'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 'OpenPorchSF','GarageArea',
    ]

df_train[fill_zero_cols] = df_train[fill_zero_cols].fillna(0)
df_test[fill_zero_cols] = df_test[fill_zero_cols].fillna(0)

df_train,df_val,y_train,y_val = train_test_split(df_train,y_train,train_size=0.8,random_state=42)

OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)

def fill_empty(df,is_train):
    number_cols = df.select_dtypes(include=['number']).columns
    df[number_cols] = df[number_cols].fillna(df[number_cols].mode().iloc[0])

    text_cols = df.select_dtypes(include=['object']).columns

    for col in text_cols:
        most_frequent = df[col].dropna().mode()[0]
        df[col] = df[col].fillna(most_frequent)

    if is_train:
        OH_array = OH_encoder.fit_transform(df[text_cols])
    else:
        OH_array = OH_encoder.transform(df[text_cols])

    OH_cols = pd.DataFrame(OH_array)
    OH_cols.index = df.index
    num_X_train = df.drop(text_cols, axis=1)
    df = pd.concat([num_X_train, OH_cols], axis=1)
    df.columns = df.columns.astype(str)
    return df

threshold = 0.6 * len(df_train)
original_cols = df_train.columns
df_train = df_train.dropna(thresh=threshold,axis=1)
empty_cols = list(set(original_cols)-set(df_train.columns))

df_train = fill_empty(df_train,True)

mi_scores = mutual_info_regression(df_train, y_train)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=df_train.columns)
mi_scores = mi_scores.sort_values(ascending=False)
useful_features = mi_scores[mi_scores > 0.01].index
df_train_cleaned = df_train[useful_features]

df_val = df_val.drop(empty_cols,axis=1)
df_val = fill_empty(df_val,False)
df_val_cleaned = df_val[useful_features]
df_test = df_test.drop(empty_cols,axis=1)
df_test = fill_empty(df_test,False)
df_test_cleaned = df_test[useful_features]

# model = XGBRegressor(n_estimators=5000,learning_rate=0.01,early_stopping_rounds=12,random_state=50)
# model.fit(df_train_cleaned,y_train,verbose=False,eval_set=[(df_val_cleaned,y_val)])

# predictions = model.predict(df_test_cleaned)

param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3,10),
    'subsample': uniform(0.6, 0.4),
    'early_stopping_rounds': randint(3,10)
}

xgbr = XGBRegressor(random_state=42,n_jobs=1)
random_search = RandomizedSearchCV(estimator=xgbr,param_distributions=param_dist,n_iter=20,cv=5,n_jobs=-1)
random_search.fit(df_train_cleaned,y_train,eval_set=[(df_val_cleaned, y_val)],verbose=False)
predictions = random_search.predict(df_test_cleaned)
sub = pd.DataFrame(predictions,columns=['SalePrice'])
ids = [i for i in range(1461,len(predictions)+1461)]
sub['Id'] = ids
sub = sub[['Id','SalePrice']]
sub.to_csv('pred.csv',index=False)