mkdir -p $1
python3 train.py \
--experiment_name $4 \
--accumulate_metrics $5 \
--layers "128,128" \
--save_path=$1 \
--ntopics $2 \
--epoch 128 \
--beta1 0.0 \
--beta2 0.99 \
--batch_size 256 \
--lr "8e-3,127" \
--temp "0.7,127" \
--train_data "./data/TMN/title/TMNtitle.npy" \
--test_data "./data/TMN/title/TMNtitle.npy" \
--label_test "./data/TMN/title/TMNtitle.LABEL" \
--word2idx "./data/TMN/word2idx.json" \
--word_vec "./data/TMN/$3_wv.npy" \
--alpha 0.1 \
--burn_in 8128
