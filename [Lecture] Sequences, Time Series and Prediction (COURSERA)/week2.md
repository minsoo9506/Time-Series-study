## Preparing features and labels
- [1,2,3,4,5] 라는 시계열 데이터가 있을때
- x = [1,2,3], y = [4] 이런 형태로 데이터를 가공한다. (window size=3)
- batch
- suffle (for sequence bias treat)

## Feeding windowed dataset into NN
- fully-connetected layer를 이용한다.
- learning rate를 decay해본다.