# 等预算数据训练对照

Two-source matched-training diagnostic after frozen-policy probes; not a final experiment or independent generalization test

两动作×两数据组；每组512环境×24步×1000更新，随机初始化种子0，参数初值逐张量哈希匹配；固定最终model_999，不挑检查点。

|动作|训练数据|评估参考|条件|完整执行|严格通过|根误差(m)|
|---|---|---|---|---:|---:|---:|
|Horse-stance_punch|raw|raw|randomized|18.75%|0.00%|0.2458|
|Horse-stance_punch|raw|raw|nominal|100.00%|0.00%|0.2146|
|Horse-stance_punch|raw|quality_gated|randomized|25.00%|0.00%|0.2547|
|Horse-stance_punch|raw|quality_gated|nominal|0.00%|0.00%|0.2184|
|Horse-stance_punch|quality_gated|raw|randomized|18.75%|0.00%|0.4608|
|Horse-stance_punch|quality_gated|raw|nominal|96.88%|0.00%|0.5111|
|Horse-stance_punch|quality_gated|quality_gated|randomized|15.62%|0.00%|0.5316|
|Horse-stance_punch|quality_gated|quality_gated|nominal|0.00%|0.00%|0.4315|
|Side_kick|raw|raw|randomized|6.25%|0.00%|0.3250|
|Side_kick|raw|raw|nominal|0.00%|0.00%|0.1896|
|Side_kick|raw|quality_gated|randomized|0.00%|0.00%|0.3443|
|Side_kick|raw|quality_gated|nominal|0.00%|0.00%|0.2088|
|Side_kick|quality_gated|raw|randomized|0.00%|0.00%|0.3205|
|Side_kick|quality_gated|raw|nominal|0.00%|0.00%|0.1476|
|Side_kick|quality_gated|quality_gated|randomized|0.00%|0.00%|0.2986|
|Side_kick|quality_gated|quality_gated|nominal|0.00%|0.00%|0.1509|

主比较为同一未修改原始参考、同一随机扰动条件下的严格通过率。均值包含首次终止帧，必须结合完成率解释。没有独立动作测试、多种子或双人格斗收益结论。
