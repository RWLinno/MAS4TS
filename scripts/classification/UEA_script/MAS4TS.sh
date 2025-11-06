export CUDA_VISIBLE_DEVICES=0

model_name=MAS4TS

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --des 'MAS4TS_Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --des 'MAS4TS_Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --des 'MAS4TS_Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10
