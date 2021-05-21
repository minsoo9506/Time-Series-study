import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def modelCatboost(X_train_cat, y_train_cat ,X_val_cat, y_val_cat, X_test_cat, category_cols, weight=None, targetWeight=None,
    params=None, selected_feature=None):
    
    if selected_feature:
        X_train_cat, X_val_cat, X_test_cat = \
        X_train_cat[selected_feature], X_val_cat[selected_feature], X_test_cat[selected_feature]

    if weight:
        sample_weight = np.zeros(X_train_cat.shape[0])    
        for idx in weight:    
            sample_weight[idx:] = weight[idx]
    
    if targetWeight:
        sample_weight = np.zeros(X_train_cat.shape[0])
        tmp_idx = X_train_cat.loc[X_train_cat['매수고객수']!=0].index
        sample_weight[tmp_idx] = targetWeight
    
    if not params:
        params = {
            'iterations': 5000,
            'learning_rate': 0.05,
            'random_seed': 42,
            'use_best_model': True,
            'task_type' : 'GPU',
            'early_stopping_rounds' : 500,
            'eval_metric' : 'RMSE'
        }

    train_pool = Pool(X_train_cat, 
                      y_train_cat, 
                      cat_features=category_cols, 
                      weight=sample_weight)
    validate_pool = Pool(X_val_cat, 
                         y_val_cat, 
                         cat_features=category_cols)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=validate_pool, verbose=100)
    
    pred_train = model.predict(X_train_cat)
    pred_val = model.predict(X_val_cat)
    pred_test = model.predict(X_test_cat)
    
    # model score
    print('catboost best score')
    print(model.best_score_)
        
    return model, pred_train, pred_val, pred_test

def modelLightgbm(X_train, y_train ,X_val, y_val, X_test, category_cols, weight=None, params=None, selected_feature=None):
    
    if selected_feature:
        X_train, X_val, X_test = \
        X_train[selected_feature], X_val[selected_feature], X_test[selected_feature]
        
    sample_weight = np.zeros(X_train.shape[0])    
    for idx in weight:    
        sample_weight[idx:] = weight[idx]
    
    if not params:
        params = {'objective': 'regression',
                     'metric': 'rmse',
                     'boosting_type': 'gbdt',
                     'learning_rate': 0.005,
                     'seed': 42,
                     'num_iterations' : 5000,
                     'early_stopping_rounds' : 1000
                    }

    trn_data = lgb.Dataset(X_train,
                           label=y_train,
                           categorical_feature=category_cols, 
                           weight=sample_weight)
    val_data = lgb.Dataset(X_val,
                           label=y_val,
                           categorical_feature=category_cols)

    model = lgb.train(params,
                    trn_data,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100)
    
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
        
    return model, pred_train, pred_val, pred_test

def linear(X_train, y_train, X_val, y_val, X_test, use_ridge=False, use_lasso=False, max_iter=1000):
    
    if use_ridge:
        ridge = Ridge(max_iter=max_iter)
        ridge.fit(X_train, y_train)
        
        pred_train = ridge.predict(X_train)
        pred_val = ridge.predict(X_val)
        pred_test = ridge.predict(X_test)
        
        print(f'ridge train rmse : {np.sqrt(mean_squared_error(y_train, pred_train))}')
        print(f'ridge validation rmse : {np.sqrt(mean_squared_error(y_val, pred_val))}')
        
        return ridge, pred_train, pred_val, pred_test

    if use_lasso:
        lasso = Lasso(max_iter=max_iter)
        lasso.fit(X_train, y_train)
        
        pred_train = lasso.predict(X_train)
        pred_val = lasso.predict(X_val)
        pred_test = lasso.predict(X_test)
        
        print(f'lasso train rmse : {np.sqrt(mean_squared_error(y_train, pred_train))}')
        print(f'lasso validation rmse : {np.sqrt(mean_squared_error(y_val, pred_val))}')
        
        return lasso, pred_train, pred_val, pred_test