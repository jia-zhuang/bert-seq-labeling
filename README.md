# bert-seq-labeling

使用原生 [google-research/bert](https://github.com/google-research/bert) 实现序列标注。提供了一个完整例子，具备一下特点：

### 依赖简单

- tensorflow 1.14.0

- 项目中的 `modeling.py` 和 `optimization.py` 来自 google-research/bert

### 支持层裁剪

- `bert.py`是对`modeling.py`的微小修改，可支持只使用指定层，从而缩小模型规模，节约显存
- 使用方法

```python
# 原生 bert
model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
)

# 支持层裁剪的 bert


model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    layer_list=[0, 1, 3, 5, 7, 9, 11],  # 层裁剪
)
```

- 裁剪会导致指标轻微降低，视不同任务而定

- 根据经验，保留 `[0, 1, 3, 5, 7, 9, 11]` 层可以获得较好结果，针对不同任务，可以尝试不同裁剪策略

### TF Estimator

基于 tensorflow 高阶 api，`Estimator`

### TF Serving

- ckpt 转 saved model：`convert_ckpt_to_saved_model.py`

为了方便计算在 dev 集上的 metric，使用了 `estimator.predict` 方法获取预测指和真值，因此在导出模型时需要把逻辑矫正过来

```python
# 转化前代码做如下调整

# elif mode == tf.estimator.ModeKeys.PREDICT:
elif mode == tf.estimator.ModeKeys.EVAL:
```


- 使用 docker 起 tf-serving 服务：`tf_serving.sh`

- 测试 tf-serving：`test_tf_serving.py`

### 训练/评估脚本

- `run.sh`
