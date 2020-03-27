# bert-seq-labeling

使用原生 [google-research/bert](https://github.com/google-research/bert) 实现序列标注。提供了一个完整例子，具备一下特点：

### 依赖简单

- tensorflow 1.14.0

- 项目中的 `modeling.py` 和 `optimization.py` 来自 google-research/bert

### TF Estimator

基于 tensorflow 高阶 api，`Estimator`

### TF Serving

- ckpt 转 saved model：`convert_ckpt_to_saved_model.py`

- 使用 docker 起 tf-serving 服务：`tf_serving.sh`

- 测试 tf-serving：`test_tf_serving.py`

### 训练/评估脚本

- `run.sh`