# Lecture
- Yonsei Univ. Time Series Analysis (undergraduate class)
- Udemy lecture
    - time series with pandas
    - time series with statsmodels
    - general model (ARIMA)
    - SARIMAX, VAR
    - Deep learning (keras)
    - Prophet
- (COURSERA) Sequences, Time Series and Prediction

# Book
- Time Series Analysis and Its Applications (Robert H.Shumway, Dvid S.Stoffer)
- forecasting : principles and practice
    - time series graphics
    - time series regression models
    - time series decomposition
    - exponential smoothing
    - ARIMA models
- 실전 시계열 분석

# Paper
### Survey
- Time Series Forecasting With Deep Learning: A Survey (2020)
    - [`Paper Link`](https://arxiv.org/abs/2004.13408) | `My Summary` | `My Code`
- An Experimental Review on Deep Learning Architecture for Time Series Forecasting (2020)
    - [`Paper Link`](https://arxiv.org/abs/2103.12057) | `My Summary` | `My Code`
### Generative model
- Time-series Generative Adversarial Networks
    - [`Paper Link`](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) | [`My summary`]((https://minsoo9506.github.io/blog/TimeGAN/)) | `My Code`
### Forecasting
- key word : `CNN`, `RNN`, `Attention`, `Multi-horizon(iterative, direct)`, `hybrid(probabilistic, non probabilistic)`, `Interpretation`, `Causal`
- Conditional Time Series Forecasting with Convolutional Neural Networks (arxiv 2018)
  - [`Paper Link`](https://arxiv.org/abs/1703.04691) | `My summary` | [`My Code`](./[[Project]%20논문%20구현])  
- DeepAR : Probabilistic Forecasting with Autoregressive Recurrent Networks (2017)
  - [`Paper Link`](https://arxiv.org/abs/1704.04110) | `My summary` | [`My Code`](./[[Project]%20논문%20구현])
- A Hybrid CNN-LSTM Model for Forecasting Particulate Matter (2020)
  - [`Paper Link`](https://ieeexplore.ieee.org/abstract/document/8979420) | `My summary` | `My Code`
- Probabilistic Time Series Forecasting with Structured Shape and Temporal Diversity (NIPS 2020)
  - [`Paper Link`](https://proceedings.neurips.cc//paper/2020/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf) | `My summary` | `My Code`
- N-BEATS : Neural Basis Expansion Analysis for Interpretable Time Series Forecasting (ICLR 2020)
  - [`Paper Link`](https://iclr.cc/virtual_2020/poster_r1ecqn4YwB.html) | `My summary` | `My Code`

### Uncertainty
- Deep and Confident Prediction for Time Series at Uber (2017)
  - [`Paper Link`](https://arxiv.org/abs/1709.01907) | `My summary` | `My Code`
- Deep Uncertainty Quantification : A Machine Learning Approach for Weather Forecasting (2018)
  - [`Paper Link`](https://arxiv.org/abs/1812.09467) | `My summary` | `My Code`

### Representation Learning
- Unsupervised Scalable Representation Learning for Multivariate Time Series (NIPS 2019)
  - [`Paper Link`](https://arxiv.org/abs/1901.10738) | `My summary` | `My Code`
- Learning Representations for Time Series Clustering (NIPS 2019)
  - [`Paper Link`](https://papers.nips.cc/paper/2019/hash/1359aa933b48b754a2f54adb688bfa77-Abstract.html) | `My summary` | `My Code`

### Classification
- Multivariate LSTM-FCNs for Time Series Classification (2018)
  - [`Paper Link`](https://arxiv.org/abs/1801.04503) | `My summary` | `My Code`
- Deep Neural Network Ensembles for Time Series Classification (2019)
  - [`Paper Link`](https://arxiv.org/abs/1903.06602) | `My summary` | `My Code`

# Project
- Kaggle predict future sale
  - 2020.06
  - `Arima`, `Boosting`
- 2020 금융 빅데이터 페스티벌 미래에셋대우
  - 2020.08
  - 주식거래내역으로 매수 상위종목 예측
  - `Linear model`, `Boosting`, `Neural Net` ,`BayesianOptimization`, `SHAP`
- 딥러닝 논문 구현 프로젝트 (진행중)
  - 2021.05 ~
  - data : Dacon 전력사용예측
  - `Dilated CNN`, `VanillaTransformer`, `DeepAR`