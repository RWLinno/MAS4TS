# MAS4TS 实验脚本

这个目录包含了MAS4TS在各个任务上的实验脚本。

## 目录结构

```
src/scripts/
├── long_term_forecast/
│   └── ETT_script/
│       └── MAS4TS_ETTh1.sh      # ETTh1数据集的长期预测
├── classification/
│   └── UEA_script/
│       └── MAS4TS.sh             # UEA数据集的分类任务
├── imputation/
│   └── ETT_script/
│       └── MAS4TS_ETTh1.sh      # ETTh1数据集的填补任务
└── anomaly_detection/
    ├── MSL/
    │   └── MAS4TS.sh             # MSL数据集的异常检测
    └── SMAP/
        └── MAS4TS.sh             # SMAP数据集的异常检测
```

## 使用方法

### 1. Long-term Forecasting (长期预测)

```bash
cd /data/sony/VQualA2025/rwl/MAS4TS
bash src/scripts/long_term_forecast/ETT_script/MAS4TS_ETTh1.sh
```

该脚本会依次运行4个不同预测长度的实验：
- pred_len=96
- pred_len=192
- pred_len=336
- pred_len=720

### 2. Classification (分类)

```bash
cd /data/sony/VQualA2025/rwl/MAS4TS
bash src/scripts/classification/UEA_script/MAS4TS.sh
```

该脚本会依次在3个UEA数据集上进行分类实验：
- EthanolConcentration
- FaceDetection
- Heartbeat

### 3. Imputation (填补)

```bash
cd /data/sony/VQualA2025/rwl/MAS4TS
bash src/scripts/imputation/ETT_script/MAS4TS_ETTh1.sh
```

该脚本会测试4个不同的缺失率：
- mask_rate=0.125
- mask_rate=0.25
- mask_rate=0.375
- mask_rate=0.5

### 4. Anomaly Detection (异常检测)

**MSL数据集：**
```bash
cd /data/sony/VQualA2025/rwl/MAS4TS
bash src/scripts/anomaly_detection/MSL/MAS4TS.sh
```

**SMAP数据集：**
```bash
cd /data/sony/VQualA2025/rwl/MAS4TS
bash src/scripts/anomaly_detection/SMAP/MAS4TS.sh
```

## 修改GPU设置

每个脚本开头都有：
```bash
export CUDA_VISIBLE_DEVICES=0
```

如果需要使用其他GPU，请修改数字。例如使用GPU 2：
```bash
export CUDA_VISIBLE_DEVICES=2
```

## 自定义实验

如果需要自定义参数，可以直接修改脚本或运行单个命令：

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_custom \
  --model MAS4TS \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --des 'My_Experiment' \
  --itr 1
```

## 结果保存

所有实验结果会保存在：
- `./checkpoints/` - 模型检查点
- `./logs/` - 训练日志
- `./results/` - 预测结果和可视化

## 注意事项

1. **MAS4TS是独立模型**：不需要指定`--base_model`参数
2. **并发执行**：多个agents会自动并发执行，提升效率
3. **内存占用**：相比大型LLM，MAS4TS内存占用更低
4. **首次运行**：agents会自动初始化，可能需要额外时间

## 批量实验

如果需要运行所有实验：

```bash
cd /data/sony/VQualA2025/rwl/MAS4TS

# 长期预测
bash src/scripts/long_term_forecast/ETT_script/MAS4TS_ETTh1.sh

# 分类
bash src/scripts/classification/UEA_script/MAS4TS.sh

# 填补
bash src/scripts/imputation/ETT_script/MAS4TS_ETTh1.sh

# 异常检测
bash src/scripts/anomaly_detection/MSL/MAS4TS.sh
bash src/scripts/anomaly_detection/SMAP/MAS4TS.sh
```

或者创建一个批量运行脚本：

```bash
#!/bin/bash
cd /data/sony/VQualA2025/rwl/MAS4TS

echo "Running Long-term Forecasting..."
bash src/scripts/long_term_forecast/ETT_script/MAS4TS_ETTh1.sh

echo "Running Classification..."
bash src/scripts/classification/UEA_script/MAS4TS.sh

echo "Running Imputation..."
bash src/scripts/imputation/ETT_script/MAS4TS_ETTh1.sh

echo "Running Anomaly Detection..."
bash src/scripts/anomaly_detection/MSL/MAS4TS.sh
bash src/scripts/anomaly_detection/SMAP/MAS4TS.sh

echo "All experiments completed!"
```

## 问题排查

如果遇到问题：

1. **检查数据集路径**：确保`./dataset/`目录下有相应数据
2. **检查GPU**：`nvidia-smi`查看GPU状态
3. **查看日志**：检查`./logs/`目录下的日志文件
4. **减小batch_size**：如果内存不足，可以减小batch_size参数

