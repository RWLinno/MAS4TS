export CUDA_VISIBLE_DEVICES=0

model_name=MAS4TS

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP/ \
  --model_id SMAP \
  --model $model_name \
  --data SMAP \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --d_model 128 \
  --d_ff 128 \
  --anomaly_ratio 1 \
  --des 'MAS4TS_Exp' \
  --itr 1

