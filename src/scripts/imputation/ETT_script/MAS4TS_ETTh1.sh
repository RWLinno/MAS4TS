export CUDA_VISIBLE_DEVICES=0

model_name=MAS4TS

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.125 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --mask_rate 0.125 \
  --des 'MAS4TS_Exp' \
  --itr 1

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.25 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --mask_rate 0.25 \
  --des 'MAS4TS_Exp' \
  --itr 1

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.375 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --mask_rate 0.375 \
  --des 'MAS4TS_Exp' \
  --itr 1

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_0.5 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --mask_rate 0.5 \
  --des 'MAS4TS_Exp' \
  --itr 1

