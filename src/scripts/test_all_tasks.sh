#!/bin/bash
# 快速测试MAS4TS所有任务是否正常运行
# 每个任务只运行1个epoch作为快速验证

cd /data/sony/VQualA2025/rwl/MAS4TS

# 激活conda环境
source /data/sony/anaconda3/etc/profile.d/conda.sh
conda activate MAS4TS

export CUDA_VISIBLE_DEVICES=0

echo "======================================"
echo "Testing MAS4TS on All Tasks"
echo "======================================"

# Test 1: Long-term Forecasting
echo ""
echo "[1/4] Testing Long-term Forecasting..."
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id test_forecast \
  --model MAS4TS \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 1 \
  --batch_size 32 \
  --des 'Test' \
  --itr 1

if [ $? -eq 0 ]; then
    echo "✓ Long-term Forecasting test PASSED"
else
    echo "✗ Long-term Forecasting test FAILED"
    exit 1
fi

# Test 2: Classification
echo ""
echo "[2/4] Testing Classification..."
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model MAS4TS \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --train_epochs 1 \
  --des 'Test' \
  --itr 1

if [ $? -eq 0 ]; then
    echo "✓ Classification test PASSED"
else
    echo "✗ Classification test FAILED"
    exit 1
fi

# Test 3: Imputation
echo ""
echo "[3/4] Testing Imputation..."
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id test_imputation \
  --model MAS4TS \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --mask_rate 0.25 \
  --train_epochs 1 \
  --batch_size 32 \
  --des 'Test' \
  --itr 1

if [ $? -eq 0 ]; then
    echo "✓ Imputation test PASSED"
else
    echo "✗ Imputation test FAILED"
    exit 1
fi

# Test 4: Anomaly Detection
echo ""
echo "[4/4] Testing Anomaly Detection..."
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL/ \
  --model_id test_anomaly \
  --model MAS4TS \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --d_model 128 \
  --d_ff 128 \
  --anomaly_ratio 1 \
  --train_epochs 1 \
  --batch_size 32 \
  --des 'Test' \
  --itr 1

if [ $? -eq 0 ]; then
    echo "✓ Anomaly Detection test PASSED"
else
    echo "✗ Anomaly Detection test FAILED"
    exit 1
fi

echo ""
echo "======================================"
echo "All Tests PASSED! ✓"
echo "======================================"
echo ""
echo "MAS4TS is ready for full experiments."
echo "Run the scripts in src/scripts/ to start training."
