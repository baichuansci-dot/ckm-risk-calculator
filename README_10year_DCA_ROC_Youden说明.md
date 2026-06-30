# 10年DCA、IPCW time-dependent ROC Youden 与网页风险分层说明

## 1. 10年风险定义

本网页计算器输出的10年死亡风险定义为固定时间点累计事件概率：

```text
10年死亡风险 = 1 - S(120 months)
```

其中 `S(120 months)` 为生存模型估计的120个月生存概率。

## 2. 为什么使用10年而不是20年

NHANES内部验证可支持120个月（10年）风险估计，因此网页计算器改为10年风险；新下载CHARLS 2011-2020资料不足以形成严格120个月外部验证，CHARLS应作为8年外部/敏感性验证报告。20年估计不宜作为网页主要输出，也不宜表述为“20年预测已外部验证”。

## 3. Youden阈值方法

旧版阈值文件 `risk_thresholds_10_year_Youden.csv` 使用固定10年二分类近似：

- 10年内发生事件者作为病例；
- 随访超过10年者作为对照；
- 10年前删失者被排除。

这种方法方向正确，但不是最严格的time-dependent ROC，因为没有对删失进行IPCW校正。

新版阈值文件 `risk_thresholds_10_year_Youden_IPCW.csv` 使用10年IPCW time-dependent ROC的Youden指数：

```text
Youden index = sensitivity + specificity - 1
```

高风险阈值取Youden指数最大的预测风险值。

## 4. 当前IPCW Youden结果

阈值应在训练集/derivation cohort 中推导，测试集/validation cohort 用于验证ROC、DCA和校准曲线，不用于确定网页阈值。

基于训练集、最终网页模型和10年风险 `1 - S(120 months)`，当前用于网页分层的IPCW Youden阈值为：

| 结局 | 模型 | 训练集10年IPCW AUC | 高风险阈值 | 敏感度 | 特异度 |
|---|---|---:|---:|---:|---:|
| 全因死亡 | GradientBoostingSurvival | 0.8858 | 7.75% | 0.7981 | 0.7954 |
| 心血管死亡 | RandomSurvivalForest | 0.9878 | 5.47% | 0.9687 | 0.9279 |

测试集自身最佳阈值仅作对照：全因死亡约3.98%，心血管死亡约1.01%。这些测试集最佳阈值不用于网页计算器，以避免在验证集上重新选择cutoff。

## 5. 为什么只分高风险和低风险

网页计算器当前采用二分类风险分层：

- **High Risk**：预测10年风险 ≥ 10年IPCW Youden阈值；
- **Low / Non-high Risk**：预测10年风险 < 10年IPCW Youden阈值。

这样做的统计学理由：

1. 只使用一个有明确统计学来源的阈值；
2. 避免使用“高风险阈值的一半”作为低风险阈值，因为该规则缺乏独立统计学依据；
3. 更容易向审稿人解释；
4. 与DCA和ROC分析逻辑一致。

## 6. DCA曲线解释

DCA（decision curve analysis）用于评估模型在不同阈值概率下的净获益。图中包含：

- Model：使用模型判断高风险；
- Treat all：所有人都视为需要干预；
- Treat none：所有人都不干预。

如果模型曲线在临床相关阈值范围内高于 Treat all 和 Treat none，说明模型在该阈值范围内具有更好的净获益。

本项目将10年DCA重点展示在0%–30%的阈值概率区间，因为当前10年高风险阈值约为5%–8%，该区间更符合图形解释需要。

## 7. 生成文件

验证输出目录：

```text
validation_outputs_10year/
```

主要文件包括：

- `ROC_10year_all_cause_IPCW.pdf/png`
- `ROC_10year_cardiovascular_IPCW.pdf/png`
- `DCA_10year_all_cause_IPCW.pdf/png`
- `DCA_10year_cardiovascular_IPCW.pdf/png`
- `Calibration_10year_all_cause.pdf/png`
- `Calibration_10year_cardiovascular.pdf/png`
- `risk_thresholds_10_year_Youden_IPCW.csv`
- `dca_net_benefit_10year_all_cause.csv`
- `dca_net_benefit_10year_cardiovascular.csv`
- `validation_summary_10year.txt`

## 8. 推荐论文表述

英文：

> The predicted 10-year mortality risk was calculated as one minus the estimated survival probability at 120 months. The high-risk threshold was determined using the Youden index from an inverse probability of censoring weighted time-dependent ROC analysis at 10 years. Decision curve analysis was performed to evaluate the net benefit of the model across a range of clinically relevant threshold probabilities. For risk communication in the web calculator, participants were classified as high risk if their predicted 10-year risk was greater than or equal to the Youden-derived threshold, and as low/non-high risk otherwise.

中文：

> 10年预测风险定义为120个月时的累计事件概率，即1减去120个月生存概率。高风险阈值通过10年IPCW时间依赖ROC曲线的Youden指数确定。DCA用于评估模型在不同阈值概率下相对于全部干预和全部不干预策略的净获益。网页计算器中，预测风险大于或等于Youden阈值者定义为高风险，其余定义为低风险或非高风险。

保守补充：

> 风险分层主要用于风险沟通和研究展示，不应被解释为直接治疗决策阈值。
