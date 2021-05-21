import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trade = pd.read_csv('trade_train.csv', index_col=0)
stock = pd.read_csv('stocks.csv', index_col=0)
answer = pd.read_csv('answer_sheet.csv')

from makeDataSet2 import makeDataset, makeCV, encoding, makeSub
from myModel import linear
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

df_rolling23_diff12_MinMax = pd.read_csv('df_rolling23_diff12_MinMax.csv')

X_train, y_train, X_val, y_val, X_test = \
     makeCV(df_rolling23_diff12_MinMax, train_=[201911, 201912, 202001, 202002, 202003, 202004])


category_cols = ['종목번호', '시장구분', '표준산업구분코드_대분류', '표준산업구분코드_중분류', '그룹번호']
X_train_en, X_val_en, X_test_en = encoding(X_train.copy(), y_train.copy(), X_val.copy(), X_test.copy(), 
    category_cols=category_cols, year2020=True, dropTargetZero=True)

def lassoModel2020(train_cols):
    groups = X_train['그룹번호'].unique()
    X_train_en['pred_train']=np.nan
    X_val_en['pred_val']=np.nan
    X_test_en['pred_test']=np.nan
    val_error = []
    coef = np.zeros([48,len(train_cols)])

    for idx, group in enumerate(groups):

        linear_train = X_train_en.loc[X_train['그룹번호']==group, train_cols]
        linear_val = X_val_en.loc[X_val['그룹번호']==group, train_cols]
        linear_test = X_test_en.loc[X_test['그룹번호']==group, train_cols]

        lasso = Lasso(max_iter=2000)
        lasso.fit(linear_train, y_train.iloc[linear_train.index])
        
        X_train_en.loc[X_train['그룹번호']==group, 'pred_train'] = lasso.predict(linear_train)
        X_val_en.loc[X_val['그룹번호']==group, 'pred_val'] = lasso.predict(linear_val)
        X_test_en.loc[X_test['그룹번호']==group, 'pred_test'] = lasso.predict(linear_test)
        
        coef[idx] = lasso.coef_

        mse = mean_squared_error( y_val.iloc[linear_val.index], X_val_en.loc[X_val['그룹번호']==group, 'pred_val'])
        print('lasso validation rmse', group, np.sqrt(mse))
        val_error.append(mse)
    print('[coef == 0]')
    for i, t in enumerate((coef.mean(axis=0) == 0).tolist()):
        if t:
            print(train_cols[i])

    return coef, val_error


train_cols = ['거래금액_mean', '거래금액_max', '거래금액_min',
    #'거래량4_mean', '거래량4_max', '거래량4_min',
    '매수고객수',
    #'경제심리지수(전월차)(p)', '코스피(전월비)(%)',
    '매수고객수rolling_mean2',
    '매수고객수rolling_mean3', '매수고객수rolling_std2', '매수고객수rolling_std3',
    '매수고객수rolling_max2', '매수고객수rolling_max3',
    '매수고객수rolling_min2', '매수고객수rolling_min3', 
    '매수고객수diff1', '매수고객수diff2']

coef6, val_error6 = lassoModel2020(train_cols)
print(np.mean(val_error6)) # 350