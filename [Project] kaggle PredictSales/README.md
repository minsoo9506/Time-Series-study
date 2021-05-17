- 2020.06.25~2020.07.14 약 3주기간의 공부
- https://www.kaggle.com/c/competitive-data-science-predict-future-sales
- kernel과 discussion을 읽어보는데 조금 더 집중
- 기본적인 FE와 model stacking연습
- 기존 train set의 target을 뒤로 shift하여 prediction하는 방법
- 시계열 데이터이다 보니 time lag같은 시간을 이용하여 feature생성

time series data이지만 ARIMA나 deep learning을 이용한 방법은 많이 보이지는 않았다. 주로 Feature engineering을 얼마나 잘하냐의 싸움인 듯 하다. 실제 현업에서는 어떤 방법들을 사용하는지 궁금하다.

## ML
- https://www.kaggle.com/vanshjatana/applied-machine-learning
- https://www.kaggle.com/dimitreoliveira/model-stacking-feature-engineering-and-eda
## Traditional TS
- https://www.kaggle.com/jagangupta/time-series-basics-exploring-traditional-ts
    - Basic exploration/EDA
    - Single time-series
    - Stationarity
    - Seasonality , Trend and Remainder
    - AR , MA , ARMA , ARIMA
    - Selecting P and Q using AIC
    - ETS
    - Prophet
## 0.9 score pipeline
- load data
- heal data and remove outliers
- work with shops/items/cats objects and features
- create matrix as product of item/shop pairs within each month in the train set
- get monthly sales for each item/shop pair in the train set and merge it to the matrix
- clip item_cnt_month by (0,20)
- append test to the matrix, fill 34 month nans with zeros
- merge shops/items/cats to the matrix
- add target lag features
- add mean encoded features
- add price trend features
- add month
- add days
- add months since last sale/months since first sale features
- cut first year and drop columns which can not be calculated for the test set
- select best features
- set validation strategy 34 test, 33 validation, less than 33 train
- fit the model, predict and clip targets for the test set