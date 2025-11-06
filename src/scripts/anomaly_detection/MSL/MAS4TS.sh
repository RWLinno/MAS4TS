export CUDA_VISIBLE_DEVICES=0

model_name=MAS4TS

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL/ \
  --model_id MSL \
  --model $model_name \
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
  --des 'MAS4TS_Exp' \
  --itr 1

