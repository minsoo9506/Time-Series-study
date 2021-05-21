# #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trade = pd.read_csv('trade_train.csv', index_col=0)
stock = pd.read_csv('stocks.csv'
, index_col=0)
answer = pd.read_csv('answer_sheet.csv')


# #
from DataSet import makeDataset, makeCV, encoding, makeSub
from myModel import modelCatboost, modelLightgbm, linear
from sklearn.metrics import mean_squared_error


# #
best_feature = [
       '종목번호', '거래량_mean_weekdiff41',  '거래금액_mean', 
       '거래량4_mean', '그룹번호', '매수고객수', '매수고객수rolling_mean2', '매수고객수rolling_mean3', '매수고객수rolling_std2', '매수고객수rolling_std3', '매수고객수rolling_max2', '매수고객수rolling_max3', '매수고객수rolling_min2', '매수고객수rolling_min3', '매수고객수diff1', '매수고객수diff2']

# #
seq2seq = pd.read_csv('seq2seq_df.csv')

# #
#df_rolling23_diff12_MinMax.to_csv('df_rolling23_diff12_MinMax.csv', index=False)
df_rolling23_diff12_MinMax = pd.read_csv('df_rolling23_diff12_MinMax.csv')


# #
X_train_cat, y_train_cat, X_val_cat, y_val_cat, X_test_cat =     makeCV(df_rolling23_diff12_MinMax, train_=[201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004], use_catboost=True)

# # [markdown]
# # Catboost optimization

# #
X_train_cv = pd.concat([X_train_cat, X_val_cat])
y_train_cv = pd.concat([y_train_cat, y_val_cat])


# #
from bayes_opt import BayesianOptimization
from catboost import Pool, cv


# #
sample_weight = np.zeros(X_train_cv.shape[0])
sample_weight[:24864] = 1 
sample_weight[24864:] = 3

def cat_hyp(depth, bagging_temperature): # Function to optimize depth and bagging temperature
    params = {
        'iterations': 2500,
        #'learning_rate': 0.03,
        'random_seed': 42,
        'use_best_model': True,
        'task_type' : 'GPU',
        'border_count' : 254,
        'early_stopping_rounds' : 800,
        'eval_metric' : 'RMSE',
        "loss_function": "RMSE",
        'verbose' : False
    }
    params["depth"] = int(round(depth)) 
    params["bagging_temperature"] = bagging_temperature

    category_cols = ['종목번호','그룹번호','시장구분']
    cv_dataset = Pool(data=X_train_cv[best_feature],
                    label=y_train_cv,
                    cat_features=category_cols, weight=sample_weight)

    scores = cv(cv_dataset,
                params,
                fold_count=4)
    return -np.min(scores['test-RMSE-mean'])  


# #
# Search space
pds = {'depth': (8, 10),
          'bagging_temperature': (0,5)
          }

# Surrogate model
optimizer = BayesianOptimization(cat_hyp, pds, random_state=2100)
                                  
# Optimize
optimizer.maximize(init_points=3, n_iter=7)

# #
cat_answer = makeSub(X_test, pred_test_cat_fs)
cat_answer.to_csv('0921_catboost_bestFs_borderCount254_depth9_iter5000.csv', index=False)


