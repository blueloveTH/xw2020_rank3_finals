# xwbank2020_决赛rank3（来自Chaos团队）
 [“2020创青春·交子杯” 新网银行金融科技挑战赛AI算法赛道](https://www.kesci.com/custom_landing/xwbank)

## 知乎分享@Aaa

https://zhuanlan.zhihu.com/p/194353668

##  score: 0.80347

|                    | **minu/epoch** | **one fold** | **20 fold** | **acc(online)** |
| ------------------ | -------------- | ------------ | ----------- | --------------- |
| **CNN1D**          | 0.083          | 3.32         | 60          | **0.78**        |
| **CNN2D**          | 0.15           | 9            | /           | 0.76            |
| **DNN**            | **0.0125**     | **1.225**    | **22**      | 0.76            |
| **Lightgbm**       | 0.16(50step)   | 10           | /           | 0.74            |
| **GRU+Attention**  | 0.066          | 2.97         | 58          | 0.76            |
| **LSTM+Attention** | 0.08           | 3.2          | 64          | 0.75            |