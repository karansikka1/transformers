BATCH_SIZE=2
GPUS=2,3
KNOWLEDGE_METHOD=0

python train.py \
--model_name bert_swag_adamw_warmup_maxlen:150 \
--lr 5e-5 \
--batch_size $BATCH_SIZE \
--data_dir ../answer_gen/data/cosmosqa/data/ \
--val_freq 20000 \
--num_validation_samples 20006 \
--gpu_ids $GPUS \
--knowledge_method $KNOWLEDGE_METHOD \
--use_wandb \
--num_epochs 3
