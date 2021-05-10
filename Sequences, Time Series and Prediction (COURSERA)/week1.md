## Introduction to time series
- Trend
- Seasonality
- Noise

## Train, validation and test sets
- fixed partitioning
    - 시간 순서대로 train, val, test로 나눈다.
    - seasonal이 있으면 다 포함해서 나눠야 한다. 
    - train, val, test로 나눠서 다 진행한 뒤에 모두 합쳐서 다시 training.

## Metric
- error = forecast = actual
- mse = np.square(errors).mean()
    - error가 큰게 위험한 경우 (더 가중치가 크니까)
- rmse = np.sqrt(mse)
- mae = np.abs(errors).mean()
- mape = np.abs(errors/x_valid).mean()

## Moving average and differencing
- MA는 trend, seasonality가 없어야한다.
- 따라서 differencing!