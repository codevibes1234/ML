import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import uniform,randint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
y_train = df_train.pop('Survived')

df_train,df_val,y_train,y_val = train_test_split(df_train,y_train,test_size=0.2)

to_be_removed = ['PassengerId','Name','Ticket','Cabin','Fare']
ids = df_test['PassengerId']
sex_map = {'male':0,'female':1}
city_map = {'C':2,'Q':1,'S':0}

mode_imputer = SimpleImputer(strategy='most_frequent')
mean_imputer = SimpleImputer()

def preprocess_data(df):
    df = df.drop(labels=to_be_removed,axis=1)  
    df['Embarked'] = df['Embarked'].map(city_map)
    df['Sex'] = df['Sex'].map(sex_map)
    return df

param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3,10),
    'subsample': uniform(0.6, 0.4),
}

imputer = ColumnTransformer(
    transformers=[('mode_fill',mode_imputer,['Embarked']),('mean_fill',mean_imputer,['Age'])],
    remainder='passthrough')
preprocessor = FunctionTransformer(preprocess_data)

preprocessing = Pipeline(steps=[('preprocessor',preprocessor),('imputer',imputer)])

df_train = preprocessing.fit_transform(df_train,y_train)
df_test = preprocessing.transform(df_test)
df_val = preprocessing.transform(df_val)

xgbc = XGBClassifier(random_state=42,n_jobs=1)
random_search = RandomizedSearchCV(estimator=xgbc,param_distributions=param_dist,n_iter=20,cv=5,n_jobs=-1)

random_search.fit(df_train,y_train,verbose=False)

predictions = random_search.predict(df_test)
# print(accuracy_score(y_pred=predictions,y_true=y_val))

sub = pd.DataFrame(predictions,columns=['Survived'])
sub['PassengerId'] = ids
sub = sub[['PassengerId','Survived']]
sub.to_csv('pred.csv',index=False)

# print(avg_c,avg_q,avg_s)
# print(df_train)
# num_nulls = df_train.isnull().sum()
# print(num_nulls)
# num_nulls = df_test.isnull().sum()
# print(num_nulls,df_test.shape[0])
# ind = df_test[df_test['Fare'].isnull()].index.to_list()
# print(df_test.loc[ind,'Age'])