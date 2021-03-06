mkdir -p $1
python3 train.py \
--experiment_name $4 \
--accumulate_metrics $5 \
--layers "128,128" \
--save_path=$1 \
--ntopics $2 \
--epoch 256 \
--beta1 0.0 \
--beta2 0.99 \
--batch_size 256 \
--lr "8e-3,8" \
--temp "0.7,8" \
--train_data "./data/N20/short/N20short.npy" \
--test_data "./data/N20/short/N20short.npy" \
--label_test "./data/N20/short/N20short.LABEL" \
--word2idx "./data/N20/word2idx.json" \
--word_vec "./data/N20/$3_wv.npy" \
--alpha 0.1 \
--burn_in 512
