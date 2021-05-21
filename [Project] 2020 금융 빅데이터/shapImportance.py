# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trade = pd.read_csv('trade_train.csv', index_col=0)
stock = pd.read_csv('stocks.csv'
, index_col=0)
answer = pd.read_csv('answer_sheet.csv')

# 
from DataSet import makeDataset, makeCV, encoding, makeSub
from myModel import modelCatboost
from sklearn.metrics import mean_squared_error

# 
seq2seq = pd.read_csv('seq2seq_df.csv')

# 
#df_rolling23_diff12_MinMax.to_csv('df_rolling23_diff12_MinMax.csv', index=False)
df_rolling23_diff12_MinMax = pd.read_csv('df_rolling23_diff12_MinMax.csv')

# 
X_train, y_train, X_val, y_val, X_test =\
    makeCV(df_rolling23_diff12_MinMax, train_=[201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004], use_catboost=True)

# 
# catboost
params = {
    'iterations': 5000,
    #'learning_rate': 0.03,
    'random_seed': 42,
    'use_best_model': True,
    'task_type' : 'GPU',
    'border_count' : 254,
    'depth' : 8,
    'early_stopping_rounds' : 800,
    'eval_metric' : 'RMSE' 
}
category_cols = ['종목번호', '그룹번호','시장구분']
#weight = {0:1, 6192:1, 12528:1, 18912:1, 25344:2, 31776:2, 38208:3, 44640:3}
#weight = {0:1, 6144:1, 12336:1, 18528:2, 24864:2, 31248:2, 37680:3}
weight = {0:1, 6144:1, 12336:1, 18528:1, 24864:3, 31248:3, 37680:3, 44112:3}

selected_feature = ['종목번호', 
'거래금액_mean', '거래금액_max', '거래금액_min','거래량4_std', '거래량_mean_weekdiff41', '시장구분', 
 '그룹번호','그룹내고객수', '매수고객수', '매도고객수', '평균기온(°C)', '평균상대습도(%)',
       '월합강수량(00~24h만)(mm)', '최대풍속(m/s)', '경제심리지수(전월차)(p)', '코스피(전월비)(%)',
       '취업자수(전월비)(%)',  '매수고객수rolling_mean2',
       '매수고객수rolling_mean3', '매수고객수rolling_std2', '매수고객수rolling_std3',
       '매수고객수rolling_max2', '매수고객수rolling_max3',
       '매수고객수rolling_min2', '매수고객수rolling_min3', 
       '매수고객수diff1', '매수고객수diff2']

model_cat_fs, pred_train_cat_fs, pred_val_cat_fs, pred_test_cat_fs=modelCatboost(X_train, y_train, X_val, y_val, X_test,                 
    category_cols=category_cols, params=params, selected_feature=selected_feature, weight=weight,)
# 
import shap
# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model_cat_fs)
shap_values = explainer.shap_values(X_val[selected_feature])
# 
selected_feature
# 
shap.summary_plot(shap_values, X_val[selected_feature], plot_type="bar")
# 
best_feature = [
       '종목번호', '거래량_mean_weekdiff41', '시장구분', '그룹번호', '매수고객수', '매도고객수', '매수고객수rolling_mean2', '매수고객수rolling_mean3', '매수고객수rolling_std2', '매수고객수rolling_std3', '매수고객수rolling_max2', '매수고객수rolling_max3', '매수고객수rolling_min2', '매수고객수rolling_min3', '매수고객수diff1', '매수고객수diff2']


