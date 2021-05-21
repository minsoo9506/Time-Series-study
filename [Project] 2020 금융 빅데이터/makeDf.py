# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# - 각 그룹을 마다 target의 시계열 모양에 따라 다르게 생각?
#     - trade data를 통해 그룹 clustering해볼까
# %% [markdown]
# - target의 값을 quantile로 그룹핑하여 각각 모델링?
#     - 해당 모델을 통해 predict average?
#     - 근데 그러면 큰 target을 학습한 모델의 의견이 거의 따라질듯?
#     - 그러면 가중치를 좀 다르게 해서?

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

trade = pd.read_csv('trade_train.csv', index_col=0)
stock = pd.read_csv('stocks.csv', index_col=0)
answer = pd.read_csv('answer_sheet.csv')


# %%
from DataSet import makeDataset, makeCV, encoding, makeSub
from myModel import modelCatboost, modelLightgbm, linear


# %%
seq2seq_df = pd.read_csv('seq2seq_df.csv')

# %% [markdown]
# # val과 test X feature 비교 by modeling (adverserial)

# %%
# df_roling2_diff1 = makeDataset(seq2seq_df, rolling_range=[2], diff_range=[1])

# X_train_cat, y_train_cat, X_val_cat, y_val_cat, X_test_cat = \
#     makeCV(df_roling2_diff1, train_=[201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004], use_catboost=True)

# adv_val = X_val_cat.copy()
# adv_val['target'] = 1
# adv_test = X_test_cat.copy()
# adv_test['target'] = 0

# adv_train = pd.concat([adv_val, adv_test], axis=0).reset_index(drop=True)

# from catboost import CatBoostClassifier

# category_cols = ['종목번호', '시장구분', '표준산업구분코드_대분류','그룹번호']

# params = {
#     'iterations': 2000,
#     'learning_rate': 0.05,
#     'random_seed': 42,
#     'task_type' : 'GPU',
#     'early_stopping_rounds' : 500,
#     'eval_metric' : 'AUC'
# }

# train_pool = Pool(adv_train.iloc[:, :-2], 
#                   adv_train.iloc[:, -1], 
#                   cat_features=category_cols)

# model = CatBoostClassifier(**params)
# model.fit(train_pool, verbose=200)

# print('catboost feature importance')
# feature_importances = model.get_feature_importance(train_pool)
# feature_names = adv_train.iloc[:, :-2].columns
# for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
#     print('{}: {}'.format(name, score))

# %% [markdown]
# # Df 만들기

# %%
df_rolling23_diff12_NoMinMax = makeDataset(seq2seq_df, rolling_range=[2,3], diff_range=[1,2], save_na=False, use_minmax=False)

# df_rolling23_diff1_NoMinMax = makeDataset(seq2seq_df, rolling_range=[2,3], diff_range=[1], save_na=False, use_minmax=False)

# df_rolling2_diff12_NoMinmax = makeDataset(seq2seq_df, rolling_range=[2], diff_range=[1,2], save_na=False, use_minmax=False)

df_rolling2_diff1_NoMinmax = makeDataset(seq2seq_df, rolling_range=[2], diff_range=[1], save_na=False, use_minmax=False)


# %%
df_rolling23_diff12_NoMinMax.head()

# %% [markdown]
# # catboost
# %% [markdown]
# 1. SaveNA : not good
# 2. FS : only about time, 매수고객수만 해도 비슷한 성능 (매도고객수는 일단 drop)
# 3. big parameter change : grow_policy(Lossguide, Depthwise가 val rmse는 너 낮게 하는데 lb는 별로), metric(차이없음 그냥 rmse 고) 
#     - SymmetricTree : lb가 57점으로 가장 잘 나옴\
#     
# # 4. ensemble
# ## - company model 1 + time model 2 voting (company+매도는 cat, 매수는 lgb로 해볼까)
# ## - 6월에 가장 팔린애들 2개 + catboost 예측 1개 voting
# # 5. 5월도 val하지 않고 fitting

# %%
X_train_cat, y_train_cat, X_val_cat, y_val_cat, X_test_cat =     makeCV(df_rolling23_diff12_NoMinMax, train_=[201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004], use_catboost=True)

X_train, y_train, X_val, y_val, X_test =     makeCV(df_rolling23_diff12_NoMinMax, train_=[201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004], use_catboost=False)


# %%
X_train_cat.columns

# %% [markdown]
# ## cat

# %%
# catboost
params = {
    'iterations': 5000,
    'learning_rate': 0.03,
    'random_seed': 42,
    'use_best_model': True,
    'task_type' : 'GPU',
    'early_stopping_rounds' : 700,
    'eval_metric' : 'RMSE' 
}
category_cols = ['종목번호', '표준산업구분코드_대분류','그룹번호']
#weight = {0:1, 6192:1, 12528:1, 18912:1, 25344:2, 31776:2, 38208:3, 44640:3}
#weight = {0:1, 6144:1, 12336:1, 18528:2, 24864:2, 31248:2, 37680:3}
weight = {0:1, 6144:1, 12336:1, 18528:1, 24864:3, 31248:3, 37680:3, 44112:3}
selected_feature = ['종목번호', '거래금액_만원단위', '거래량1', '거래량2',
       '거래량3', '거래량4', '표준산업구분코드_대분류', '그룹번호', '그룹내고객수', '매수고객수',
       '매도고객수', '매도고객수rolling_mean2', '매도고객수rolling_mean3',
       '매도고객수rolling_std2', '매도고객수rolling_std3', '매수고객수rolling_mean2',
       '매수고객수rolling_mean3', '매수고객수rolling_std2', '매수고객수rolling_std3',
       '매도고객수diff1', '매도고객수diff2', '매수고객수diff1', '매수고객수diff2', 'group_char1','group_char4', 'year']


# %%
model_cat, pred_train_cat, pred_val_cat, pred_test_cat=    modelCatboost(X_train_cat, y_train_cat, X_val_cat, y_val_cat, X_test_cat ,                   category_cols, weight=weight, params=params, selected_feature=selected_feature)


# %%
feature_importances = model_cat.get_feature_importance()
feature_names = X_train_cat[selected_feature].columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))


# %%
cat_answer = makeSub(X_test_cat, pred_test_cat)


# %%
cat_answer.to_csv('0913_cat_fs_roll23_diff12_groupchar.csv', index=False)

# %% [markdown]
# ## lgb
# - encoding할때 최근 값만 이용?

# %%
category_cols = ['종목번호','그룹번호']
X_train, X_val, X_test = encoding(X_train, y_train, X_val, X_test, category_cols=category_cols)


# %%
X_train.columns


# %%
# lgb
params = {'objective': 'regression',
             'metric': 'rmse',
             'boosting_type': 'gbdt',
             'learning_rate': 0.01,
             'seed': 42,
             'num_iterations' : 5000,
             'early_stopping_rounds' : 500
            }
#weight = {0:1, 6192:1, 12528:1, 18912:1, 25344:2, 31776:2, 38208:3, 44640:3}
weight = {0:1, 6144:1, 12336:1, 18528:2, 24864:2, 31248:2, 37680:3}
selected_feature = ['종목번호','그룹번호', '매수고객수',
       '매수고객수rolling_mean2','매수고객수rolling_mean3', '매수고객수rolling_std2', '매수고객수rolling_std3',
       '매수고객수diff1', '매수고객수diff2']


# %%
model, pred_train, pred_val, pred_test=    modelLightgbm(X_train, y_train, X_val, y_val, X_test ,                   category_cols, weight=weight, params=params, selected_feature=selected_feature)


# %%
feature_importances = model.feature_importance()
feature_names = X_train[selected_feature].columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))


# %%


# %% [markdown]
# # ensemble
# %% [markdown]
# 일단 target pred 더해서 결과물

# %%
cat_lgb_sum = pred_test_cat + pred_test


# %%
cat_answer = makeSub(X_test_cat, cat_lgb_sum)


# %%
cat_answer.to_csv('0913_catAndLgb_fsDivide_roll23_diff12_dropNa_NoMinMax.csv', index=False)


# %%


# %% [markdown]
# # voting
# - 그룹마다 특징 이전보다 많이 사는~! 경우의 수를 count
# - 6월달 가장 잘 팔린 2개 + catboost 가장 잘 팔린 1개
# - lgb encoding : 2020년도 target만 해보기!

# %%
# 6월
month6 = seq2seq_df.loc[seq2seq_df['기준년월']==202006, ['종목번호','그룹번호','매수고객수']]

onlyMonth6 = pd.read_csv('answer_sheet.csv')
result_cols = ['종목번호','그룹번호']
sub = X_test_cat[result_cols].reset_index(drop=True)

tmp = pd.merge(X_test_cat, month6, on=['종목번호','그룹번호'], how='left')

sub['pred'] = tmp['매수고객수_y']
sub = sub.sort_values(by=['그룹번호','pred'], ascending=[True, False])
group_num = sub['그룹번호'].unique()

sub_cat = X_test_cat[result_cols].reset_index(drop=True)
sub_cat['pred'] = pred_test_cat
sub_cat = sub_cat.sort_values(by=['그룹번호','pred'], ascending=[True, False])
# sub_cat = sub_cat.sort_values(by=['그룹번호','pred'], ascending=[True, True]) # 오히려 cat이 그나마 rare한 경우 맞추도록

for num in group_num:
    val_cat = sub_cat.loc[sub_cat['그룹번호']==num][:2]['종목번호'].sort_values().values
    val = sub.loc[sub['그룹번호']==num][:3]['종목번호'].sort_values().values
    onlyMonth6.loc[onlyMonth6['그룹명']==num,'종목번호1':'종목번호2'] = val_cat
    for value in val:
        if value not in val:
            onlyMonth6.loc[onlyMonth6['그룹명']==num,'종목번호3'] = value
            break

# 최종 sort
for num in group_num:
     onlyMonth6.loc[onlyMonth6['그룹명']==num,'종목번호1':] =         pd.Series(onlyMonth6.loc[onlyMonth6['그룹명']==num,'종목번호1':].values[0]).sort_values().values


# %%
onlyMonth6.to_csv('0913_onlyMonth6big1_catbig2_voting.csv', index=False)

# %% [markdown]
# # Analysis
# %% [markdown]
# - 그룹마다 특징 이전보다 많이 사는~! 경우의 수를 count
# - 거래랑은 okay 나머지는 별로인듯
#     - 나중에 추가적으로 feauture 생성방법 생각해보기

