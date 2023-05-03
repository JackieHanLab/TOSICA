data_type=hPancreas
seed=13
n_epoch=30
learning_rate=0.001
enable_train=0
nohup python run.py \
--data_type $data_type \
--seed $seed \
--n_epoch $n_epoch \
--learning_rate $learning_rate \
--enable_train $enable_train \
--data_seed 0 \
--read_cached_prediction 0 \
--gmt_path human_gobp \
--project hGOBP_demo \
--depth 1 \
--embed_dim 48 \
--num_heads 4 \
--batch_size 8 \
> zlog/train$enable_train-$data_type-e$n_epoch-s$seed.log  2>&1 &