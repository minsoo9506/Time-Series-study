# Data
- [Dacon 전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/235736/overview/description)
- 전력 | 한국에너지공단 | 시계열 | SMAPE

# Models
- `Dilated Causal CNN`
  - model
  - lit_model (trainer)
  - dataloader
  - run
- `Transformer`
  - only model (train o, infernece x)

# tip 정리 & Reference
- `conv1d`
  - `(batch_size, feature_dim, time_seq)` 의 순서
  - Reference : https://sanghyu.tistory.com/24?category=1120072
- `Transformer` 구현
  - Reference : https://github.com/kh-kim/simple-nmt/blob/master/simple_nmt/models/transformer.py