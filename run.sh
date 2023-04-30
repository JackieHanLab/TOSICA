data_type=hPancreas
seed=4
n_epoch=30
enable_train=1
nohup python run.py \
--data_type $data_type \
--seed $seed \
--n_epoch $n_epoch \
--enable_train $enable_train \
--data_seed 0 \
> zlog/train$enable_train-$data_type-e$n_epoch-s$seed.log  2>&1 &