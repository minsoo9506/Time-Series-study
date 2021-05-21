# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trade = pd.read_csv('trade_train.csv', index_col=0)
stock = pd.read_csv('stocks.csv', index_col=0)
answer = pd.read_csv('answer_sheet.csv')


# %%
final_stock = stock.loc[stock['20년7월TOP3대상여부']=='Y'].reset_index(drop=True)
final_stock.drop(['종목명','20년7월TOP3대상여부'], axis=1, inplace=True)

final_stock['year'] = final_stock['기준일자'].map(lambda x : str(x)[:4])
final_stock['month'] = final_stock['기준일자'].map(lambda x : str(x)[4:6])
final_stock['day'] = final_stock['기준일자'].map(lambda x : str(x)[6:])

company_list = final_stock['종목번호'].unique()

def fourWeeks(x):
    if x < 8:
        return 1
    elif x < 15:
        return 2
    elif x < 22:
        return 3
    else:
        return 4

final_stock['week'] = final_stock['day'].astype(np.int16).map(fourWeeks)


# %%
final_stock['week'].value_counts()


# %%

weekby_df =  pd.DataFrame(final_stock.groupby(['year','month','종목번호','week'])['거래량'].agg(['mean', 'max', 'min', 'std']))    .unstack().reset_index()
weekby_df['기준일자'] = weekby_df['year'].map(str) + weekby_df['month'].map(str)
weekby_df.columns = ['year','month','종목번호','거래량1_mean','거래량2_mean','거래량3_mean','거래량4_mean','거래량1_max','거래량2_max','거래량3_max','거래량4_max','거래량1_min','거래량2_min','거래량3_min','거래량4_min','거래량1_std','거래량2_std','거래량3_std','거래량4_std','기준일자']

weekby_df['거래량_mean_weekdiff41'] = weekby_df['거래량4_mean'] - weekby_df['거래량1_mean']
weekby_df['거래량_mean_weekdiff42'] = weekby_df['거래량4_mean'] - weekby_df['거래량2_mean']
weekby_df['거래량_mean_weekdiff43'] = weekby_df['거래량4_mean'] - weekby_df['거래량3_mean']

weekby_df['거래량_max_weekdiff41'] = weekby_df['거래량4_max'] - weekby_df['거래량1_max']
weekby_df['거래량_max_weekdiff42'] = weekby_df['거래량4_max'] - weekby_df['거래량2_max']
weekby_df['거래량_max_weekdiff43'] = weekby_df['거래량4_max'] - weekby_df['거래량3_max']

weekby_df['거래량_min_weekdiff41'] = weekby_df['거래량4_min'] - weekby_df['거래량1_min']
weekby_df['거래량_min_weekdiff42'] = weekby_df['거래량4_min'] - weekby_df['거래량2_min']
weekby_df['거래량_min_weekdiff43'] = weekby_df['거래량4_min'] - weekby_df['거래량3_min']

fillna_func = lambda x : x.fillna(x.mean())
weekby_df = weekby_df.groupby(['종목번호','year']).apply(fillna_func)

weekby_df.index = range(weekby_df.shape[0])

weekby_df_drop_cols = ['year','month','거래량1_mean','거래량2_mean','거래량3_mean','거래량1_max','거래량2_max','거래량3_max','거래량1_min','거래량2_min','거래량3_min']
weekby_df.drop(weekby_df_drop_cols, inplace=True, axis=1)

print(weekby_df.shape)
weekby_df.head()


# %%
stock_df = pd.DataFrame(final_stock.groupby(['year','month','종목번호'])['거래금액_만원단위'].agg(['mean', 'max', 'min', 'std'])).reset_index()
stock_df['기준일자'] = stock_df['year'].map(str) + stock_df['month'].map(str)
stock_df.columns = ['year','month','종목번호','거래금액_mean','거래금액_max', '거래금액_min', '거래금액_std','기준일자']
stock_df.drop(['year','month'], inplace=True, axis=1)
print(stock_df.shape)
stock_df.head()


# %%
stock_weekby_df = pd.merge(stock_df, weekby_df, how='left', on=['종목번호','기준일자'])


# %%
cols = ['종목번호','시장구분','표준산업구분코드_대분류','표준산업구분코드_중분류']
add_df =  final_stock[cols].drop_duplicates().reset_index(drop=True)
final_company = pd.merge(stock_weekby_df,add_df, on='종목번호')

final_company.rename({'기준일자' : '기준년월'}, inplace=True, axis=1)
final_company['기준년월'] = final_company['기준년월'].astype(np.int32)

final_company.columns


# %%



# %%
com_data = final_stock.groupby(['year','month','종목번호'])['종목고가','종목저가'].agg(['mean', 'max', 'min', 'std']).reset_index()
print(com_data.shape)
com_data['기준년월'] = com_data['year'] + com_data['month']
com_data.drop(['year','month'], axis=1, inplace=True)
com_data['기준년월'] = com_data['기준년월'].astype(np.int32)
com_data.columns = ['종목번호','종목고가_mean','종목고가_max','종목고가_min','종목고가_std','종목저가_mean','종목저가_max','종목저가_min','종목저가_std', '기준년월']
com_data.head()


# %%
stockForSeq2Seq = pd.merge(com_data,final_company, on=['종목번호','기준년월'], how='left')


# %%
stockForSeq2Seq.to_csv('stockForSeq2Seq.csv', index=False)

# %% [markdown]
# # trade for seq2seq

# %%
# stockForSeq2Seq = pd.read_csv('stockForSeq2Seq.csv')


# %%
top3_trade = trade.loc[trade['종목번호'].isin(company_list)].reset_index(drop=True)
top3_trade.columns


# %%
# strange value
top3_trade.iloc[348,10] =     top3_trade.loc[(top3_trade['그룹번호']=='MAD48')&(top3_trade['종목번호']=='A035420')][1:]['매수가격_중앙값'].mean()


# %%
cols = ['기준년월','그룹번호','종목번호','매수고객수','매도고객수']
final_trade = top3_trade[cols]


# %%
# 각 종목마다 그룹 만들기
tmp_com = company_list
for i in range(0,47):
    tmp_com = np.append(tmp_com,company_list)

tmp_group = sorted(final_trade['그룹번호'].unique())
for i in range(0,134):
    tmp_group = np.append(tmp_group, sorted(final_trade['그룹번호'].unique()))

tmp_df = pd.DataFrame(tmp_com).sort_values(by=0).reset_index(drop=True)

tmp_df['그룹번호'] = tmp_group
tmp_df.columns = ['종목번호','그룹번호']

# 
mse10 = ['MAD01','MAD04','MAD09','MAD13','MAD14','MAD17','MAD18','MAD21','MAD38','MAD42','MAD44']
mse20 = ['MAD05','MAD06','MAD23','MAD35','MAD41','MAD43','MAD46']
tmp_df['group_mse'] = np.where(tmp_df['그룹번호'].isin(mse10), 2, 1)
tmp_df['group_mse'] = np.where(tmp_df['그룹번호'].isin(mse20), 3, tmp_df['group_mse'])

tmp_df.tail()


# %%
seq2seq_df = pd.merge(stockForSeq2Seq, tmp_df, on='종목번호', how='left')
print(seq2seq_df.shape)

seq2seq_df.head()


# %%
tmp_numGroup = top3_trade[['그룹번호','그룹내고객수']].drop_duplicates(subset=['그룹번호']).reset_index(drop=True)
print(tmp_numGroup.shape)


# %%
seq2seq_df = seq2seq_df.merge(tmp_numGroup, on='그룹번호', how='left')
seq2seq_df = seq2seq_df.merge(final_trade, on=['그룹번호','종목번호','기준년월'], how='left')

print(seq2seq_df.shape)


# %%
seq2seq_df.isnull().sum()


# %%
# NA
seq2seq_df['매수고객수'].fillna(0, inplace=True)
seq2seq_df['매도고객수'].fillna(0, inplace=True)

fillna_func = lambda x : x.fillna(x.mean())
seq2seq_df = seq2seq_df.groupby(['종목번호']).apply(fillna_func)


# %%
seq2seq_df.index = range(seq2seq_df.shape[0])
seq2seq_df.to_csv('seq2seq_df.csv', index=False)


# %%
print(seq2seq_df.shape)
seq2seq_df.isnull().sum()


# %%
tmp = seq2seq_df.loc[seq2seq_df['매수고객수']!=0].reset_index(drop=True)
tmp['target'] = tmp.groupby(['종목번호','그룹번호'])['매수고객수'].shift(-1).reset_index(drop=True)
tmp.dropna(inplace=True)
for i in ['종목고가_mean', '종목고가_max', '종목고가_min', '종목고가_std', '종목저가_mean',
       '종목저가_max', '종목저가_min', '종목저가_std', '기준년월', '거래금액_mean', '거래금액_max',
       '거래금액_min', '거래금액_std', '거래량4_mean', '거래량4_max', '거래량4_min', '거래량1_std',
       '거래량2_std', '거래량3_std', '거래량4_std', '거래량_mean_weekdiff41',
       '거래량_mean_weekdiff42', '거래량_mean_weekdiff43', '거래량_max_weekdiff41',
       '거래량_max_weekdiff42', '거래량_max_weekdiff43', '거래량_min_weekdiff41',
       '거래량_min_weekdiff42', '거래량_min_weekdiff43', '그룹내고객수', '매수고객수', '매도고객수']:
    corr = np.corrcoef(tmp['target'], tmp[i])[0,1]
    print(f'target corr with : {i}, {corr}')


