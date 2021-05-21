import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

def makeCV(df, train_ = [201910, 201911, 201912, 202001, 202002, 202003, 202004], val_ = 202005, test_ = 202006, use_catboost=False):
    '''
    input : df, train_ = [201910, 201911, 201912, 202001, 202002, 202003, 202004], val_ = 202005, test_ = 202006, use_catboost=False
    output : X_train, y_train, X_val, y_val, X_test
    '''
    y_train = df.loc[df['기준년월'].isin(train_), 'target'].reset_index(drop=True)
    y_val = df.loc[df['기준년월']==val_, 'target'].reset_index(drop=True)
    
    X_train = df.loc[df['기준년월'].isin(train_)].reset_index(drop=True)
    X_val = df.loc[df['기준년월']==val_].reset_index(drop=True)
    X_test = df.loc[df['기준년월']==test_].reset_index(drop=True)
    
    X_train = X_train.iloc[:, :-1]
    X_val = X_val.iloc[:, :-1]
    X_test = X_test.iloc[:, :-1]
    
    X_train['year'] = X_train['기준년월'].map(str).map(lambda x : x[:4])
    X_val['year'] = X_val['기준년월'].map(str).map(lambda x : x[:4])
    X_test['year'] = X_test['기준년월'].map(str).map(lambda x : x[:4])
    X_train['year'] = X_train['year'].map({'2019' : 0, '2020' : 1})
    X_val['year'] = X_val['year'].map({'2019' : 0, '2020' : 1})
    X_test['year'] = X_test['year'].map({'2019' : 0, '2020' : 1})

    weight_idx = []
    for i in train_:
        idx = X_train.loc[X_train['기준년월']==i].index[0]
        weight_idx.append(idx)
    print(f'weight_idx : {weight_idx}')
        
    X_train.drop('기준년월', axis=1 , inplace=True)
    X_val.drop('기준년월', axis=1 , inplace=True)
    X_test.drop('기준년월', axis=1 ,inplace=True)
    
    if use_catboost:
        X_train_cat, y_train_cat = X_train.copy(), y_train.copy()
        X_val_cat, y_val_cat = X_val.copy(), y_val.copy()
        X_test_cat = X_test.copy()
        return X_train_cat, y_train_cat, X_val_cat, y_val_cat, X_test_cat
    
    return X_train, y_train, X_val, y_val, X_test

def encoding(X_train, y_train, X_val, X_test, category_cols = ['종목번호', '그룹번호'], year2020=True, dropTargetZero=True):
    '''
    input : X_train, X_val, X_test, category_cols = ['종목번호','그룹번호']
    output : X_train, X_val, X_test
    '''
    X_train_transform = X_train.copy()
    if dropTargetZero:
        X = pd.concat([X_train,y_train], axis=1)
        X = X.loc[X['target']!=0].reset_index(drop=True)
        X_train = X.iloc[:, :-1]
        y_train = X.iloc[:, -1]
        print('train len:',X_train.shape[0])

    if year2020:
        idx = X_train.loc[X_train['year']==1, category_cols].shape[0]
        encoder = ce.TargetEncoder()
        encoder.fit(X_train.loc[X_train['year']==1, category_cols], y_train[-idx:])
    else:
        idx = X_train[category_cols].shape[0]
        encoder = ce.TargetEncoder()
        encoder.fit(X_train[category_cols], y_train[-idx:])

    X_train[category_cols] = encoder.transform(X_train_transform[category_cols])
    X_val[category_cols] = encoder.transform(X_val[category_cols])
    X_test[category_cols] = encoder.transform(X_test[category_cols])
    
    return X_train, X_val, X_test

def makeSub(X_test_cat, prediction):
    '''
    input : X_test_cat, prediction
    output : answer
    '''
    answer = pd.read_csv('answer_sheet.csv')
    result_cols = ['종목번호','그룹번호']
    sub = X_test_cat[result_cols].reset_index(drop=True)
    sub['pred'] = prediction
    sub = sub.sort_values(by=['그룹번호','pred'], ascending=[True, False])
    group_num = sub['그룹번호'].unique()
    for num in group_num:
        val = sub.loc[sub['그룹번호']==num][:3]['종목번호'].sort_values().values
        answer.loc[answer['그룹명']==num,'종목번호1':] = val
    return answer


def makeDataset(before_df, use_cut=False, cut_quantile=0.99, rolling_range=[2,3], diff_range=[1,2], save_na=False, use_minmax=True):
    '''
    input : before_df, use_cut=False, cut_quantile=0.99, rolling_range=[2,3], diff_range=[1,2]
    output : df
    '''
    df = before_df.copy()
    df = df.sort_values(by=['기준년월','종목번호','그룹번호']).reset_index(drop=True)
    
    # cut
    if use_cut:
        cut1 = df['매수고객수'].quantile(cut_quantile)
        cut2 = df['매도고객수'].quantile(cut_quantile)

        df.loc[df['매수고객수'] > cut1] = cut1
        df.loc[df['매도고객수'] > cut2] = cut2

    # rolling
    rolling_cols = ['매도고객수', '매수고객수']
    for rolling_col in rolling_cols:
        for i in rolling_range:
            df[str(rolling_col)+'rolling_mean'+str(i)] = \
                df.groupby(['그룹번호','종목번호'])[rolling_col].transform(lambda x : x.rolling(i).mean())
        for i in rolling_range:
            df[str(rolling_col)+'rolling_std'+str(i)] = \
                df.groupby(['그룹번호','종목번호'])[rolling_col].transform(lambda x : x.rolling(i).std())
    if use_minmax:
        for rolling_col in rolling_cols:
            for i in rolling_range:
                df[str(rolling_col)+'rolling_max'+str(i)] = \
                    df.groupby(['그룹번호','종목번호'])[rolling_col].transform(lambda x : x.rolling(i).max())
            for i in rolling_range:
                df[str(rolling_col)+'rolling_min'+str(i)] = \
                    df.groupby(['그룹번호','종목번호'])[rolling_col].transform(lambda x : x.rolling(i).min())

    # diff
    diff_cols = ['매도고객수', '매수고객수']
    for diff_col in diff_cols:
        for i in diff_range:
            df[str(diff_col)+'diff'+str(i)] = df.groupby(['그룹번호','종목번호'])[diff_col].diff(i)
    
    if save_na:
        df = df.reset_index(drop=True)
    else:
        df = df.dropna(axis=0).reset_index(drop=True)
        
    # group char
    group_char = df[df ['매수고객수'] - \
                            df.groupby(['종목번호','그룹번호'])['매수고객수'].shift(-1) < 0]['그룹번호'].value_counts()

    group_char1 = group_char[:12].index
    group_char4 = group_char[36:48].index
    
    df['group_char1'] = np.nan
    df['group_char1'] = np.where(df['그룹번호'].isin(group_char1), 1, 0)
    # df['group_char2'] = np.nan
    # df['group_char2'] = np.where(df['그룹번호'].isin(group_char2), 1, 0)
    # df['group_char3'] = np.nan
    # df['group_char3'] = np.where(df['그룹번호'].isin(group_char3), 1, 0)
    df['group_char4'] = np.nan
    df['group_char4'] = np.where(df['그룹번호'].isin(group_char4), 1, 0)                                         

    # target
    df['target'] = df.groupby(['종목번호','그룹번호'])['매수고객수'].shift(-1)


    df_cols = df.columns
    for cols in df_cols:
        if df[cols].dtypes == np.float64:
            df[cols] = round(df[cols], 2)
    
    print(f'cut use : {use_cut}, quantile : {cut_quantile}, rolling range: {rolling_range}, diff range : {diff_range}, use MinMAx : {use_minmax}')         
    return df






