import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from scipy.stats import uniform,randint
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import mutual_info_regression

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# starboard has better survival chance

map_list = []

def preprocess_data(df,is_train=False):
    def convert(pass_id:str):
        grp,_ = pass_id.split(sep='_')
        return int(grp)

    df = df.drop('Name',axis=1)

    df['Group'] = pd.Series(convert(df.loc[i,'PassengerId']) for i in range(len(df)))
    ids = df['PassengerId']
    df.drop('PassengerId',axis=1,inplace=True)

    zero_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    for col in zero_cols:
        df[col].fillna(0,inplace=True)
    
    if is_train:
        df = df[df['Age'].notnull()]
        df.reset_index(drop=True,inplace=True)
    else:
        df['Age'].fillna(df['Age'].mean(),inplace=True)

    df['TotalCost'] = df.loc[:,'RoomService':'VRDeck'].sum(axis=1)

    for idx in df.index.to_list():
        if df.loc[idx,'Age'] >= 0 and df.loc[idx,'Age'] < 5:
            df.loc[idx,'Age'] = 3
        elif df.loc[idx,'Age'] >= 5 and df.loc[idx,'Age'] < 65:
            df.loc[idx,'Age'] = 2
        elif df.loc[idx,'Age'] >= 65 and df.loc[idx,'Age'] < 75:
            df.loc[idx,'Age'] = 1
        else:
            df.loc[idx,'Age'] = 0

    family_cols = ['HomePlanet','CryoSleep','Destination','VIP']

    for idx in range(len(df)):
        if pd.isnull(df.loc[idx,'CryoSleep']):
            if df.loc[idx,'RoomService':'VRDeck'].sum() > 0:
                df.loc[idx,'CryoSleep'] = False

    for col in family_cols:
        for row in range(len(df)):
            if pd.isnull(df.loc[row,col]):
                df_fam = df[df['Group'] == df.loc[row,'Group']]
                if df_fam[col].isnull().sum() != len(df_fam):
                    df.loc[row,col] = df_fam[col].dropna().mode()[0]
                else:
                    df.loc[row,col] = df[col].dropna().mode()[0]

    # null_rows = df[df['Cabin'].isnull()].index.to_list()
    def get_side(cabin:str):
        _,_,side = cabin.split(sep='/')
        return side
    
    if is_train:
        df = df[df['Cabin'].notnull()]
        df.reset_index(drop=True,inplace=True)
        df['Side'] = pd.Series(get_side(df.loc[idx,'Cabin']) for idx in df.index.to_list())
    else:
        side_lst = []
        for idx in range(len(df)):
            if pd.isnull(df.loc[idx,'Cabin']):
                df_fam = df[df['Group'] == df.loc[idx,'Group']]
                if df_fam['Cabin'].isnull().sum() != len(df_fam):
                    df_fam = df_fam[df_fam['Cabin'].notnull()]
                    side_lst.append(pd.Series(get_side(df_fam.loc[idx,'Cabin']) for idx in df_fam.index.to_list()).mode()[0])
                else:
                    side_lst.append('S')
            else:
                side_lst.append(get_side(df.loc[idx,'Cabin']))
        df['Side'] = pd.Series(side_lst)

    df['FamilySize'] = pd.Series(len(df[df['Group'] == df.loc[idx,'Group']]) for idx in df.index.to_list())

    # def get_deck(cabin:str):
    #     deck,_,_ = cabin.split(sep='/')
    #     return deck

    # df['Deck'] = pd.Series(get_deck(df.loc[idx,'Cabin']) for idx in df.index.to_list())
    df.drop('Cabin',axis=1,inplace=True)
    df.drop('Group',axis=1,inplace=True)

    bool_cols = ['CryoSleep','VIP']
    if is_train:
        bool_cols.append('Transported')
    for col in bool_cols:
        df[col] = df[col].astype(int)

    categorical_cols = ['HomePlanet','Destination','Side']
    if is_train:
        for col in categorical_cols:
            unique_values = df[col].unique()
            kvmap = {}
            for val in unique_values:
                kvmap[val] = df.loc[df[col] == val,'Transported'].mean()
            kvmap = dict(sorted(kvmap.items(),key = lambda item : item[1]))
            kvmap_new = {}
            idx = 0
            for k,v in kvmap.items():
                kvmap_new[k] = idx
                idx += 1
            df[col] = df[col].map(kvmap_new)
            map_list.append(kvmap_new)
        y = df.pop("Transported")
        return df,y
    else:
        idx = 0
        for col in categorical_cols:
            df[col] = df[col].map(map_list[idx])
            idx += 1
        return df,ids

param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3,10),
    'subsample': uniform(0.6, 0.4)
}

df_train,y_train = preprocess_data(df_train,True)
df_train,df_val,y_train,y_val = train_test_split(df_train,y_train,train_size=0.8)
# age = 0
# while age < 100:
#     df = df_train[(df_train['Age'] >= age) & (df_train['Age'] < age+20)]
#     print(y_train[df.index].mean())
#     age += 5

# print(df_train)
# print(pd.Series(mutual_info_regression(df_train,y_train),index=df_train.columns))

# test_size = len(df_test)
# df_test,ids = preprocess_data(df_test)


xgbr = XGBClassifier(random_state=42,n_jobs=1)
random_search = RandomizedSearchCV(estimator=xgbr,param_distributions=param_dist,n_iter=20,cv=5,n_jobs=-1)
random_search.fit(df_train,y_train,verbose=False)

predictions = random_search.predict(df_val)

# def find_pred():
#     cnt = 0
#     preds = []
#     for idx in range(test_size):
#         if idx in null_rows:
#             preds.append(1)
#         else:
#             preds.append(predictions[cnt])
#             cnt += 1
#     return preds

# sub = pd.DataFrame(pd.Series(predictions),columns = ['Transported'])
# sub['PassengerId'] = ids
# sub = sub[['PassengerId','Transported']]
# sub['Transported'] = sub['Transported'].astype(bool)
# sub.to_csv('pred.csv',index=False)

# print(df_train.isnull().sum())
# clf = LogisticRegressionCV(cv=5,solver='saga',penalty='elasticnet',max_iter = 10000, n_jobs = -1, l1_ratios=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
# clf.fit(df_train,y_train)
# predictions = clf.predict(df_train)
print(accuracy_score(y_pred=predictions,y_true=y_val))