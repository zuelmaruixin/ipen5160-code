# Aspect Network Summary

- Prediction source: `/Users/maruixin/Downloads/model_predictions.csv`
- Split filter: `('val', 'test')`
- Review count used: 9880
- Node count: 18
- Edge count retained: 138

## Top Negative Aspects

| aspect_name | negative_review_count | negative_rate_given_mentioned |
| --- | --- | --- |
| Price#Level | 1873 | 0.29468219005663937 |
| Food#Portion | 1787 | 0.30778504994832934 |
| Service#Hospitality | 1284 | 0.18597914252607184 |
| Food#Appearance | 922 | 0.21137093076570382 |
| Service#Timely | 907 | 0.4557788944723618 |

## Strongest Co-Negative Links

| source | target | co_negative_count | conditional_negative_mean | lift_mean |
| --- | --- | --- | --- | --- |
| Food#Portion | Price#Level | 675 | 0.3690562229257935 | 1.9925002636649398 |
| Food#Portion | Service#Hospitality | 472 | 0.31586553631541053 | 2.03240084584582 |
| Price#Cost_effective | Price#Level | 447 | 0.47019070472659064 | 3.7015810061344343 |
| Food#Appearance | Food#Portion | 429 | 0.35267999664970073 | 2.57252001985902 |
| Price#Level | Service#Hospitality | 413 | 0.27107647950129155 | 1.6966966217755846 |
| Service#Hospitality | Service#Timely | 401 | 0.37721108237419587 | 3.401958460846239 |
| Food#Appearance | Service#Hospitality | 368 | 0.34286834120596565 | 3.071205087139565 |
| Food#Portion | Price#Cost_effective | 351 | 0.3737194933933282 | 3.0464922398730057 |

## Strongest Directional Negative Risks

| aspect_a | aspect_b | support_ab_negative | p_b_negative_given_a_negative | lift |
| --- | --- | --- | --- | --- |
| Food#Portion | Price#Level | 675 | 0.37772803581421377 | 1.9925002636649396 |
| Price#Level | Food#Portion | 675 | 0.3603844100373732 | 1.9925002636649398 |
| Service#Hospitality | Food#Portion | 472 | 0.367601246105919 | 2.0324008458458196 |
| Food#Portion | Service#Hospitality | 472 | 0.2641298265249021 | 2.03240084584582 |
| Price#Cost_effective | Price#Level | 447 | 0.7017268445839875 | 3.7015810061344347 |
| Price#Level | Price#Cost_effective | 447 | 0.2386545648691938 | 3.7015810061344343 |
| Food#Appearance | Food#Portion | 429 | 0.46529284164859 | 2.5725200198590206 |
| Food#Portion | Food#Appearance | 429 | 0.2400671516508114 | 2.57252001985902 |

## Interpretation

这份网络更适合解释问题联动和共负面模式，不应直接解释成因果或时间转移。`co_negative_count` 回答哪些问题容易一起被抱怨，`p_b_negative_given_a_negative` 与 `lift` 回答当 A 负面时，B 是否更容易一起出问题。

