mkdir -p $1
python3 train.py \
--experiment_name $4 \
--accumulate_metrics $5 \
--layers "128,128" \
--save_path=$1 \
--ntopics $2 \
--epoch 64 \
--beta1 0.0 \
--beta2 0.99 \
--batch_size 256 \
--lr "8e-3,49" \
--temp "0.7,49" \
--train_data "./data/snippets/snippets.npy" \
--test_data "./data/snippets/snippets.npy" \
--label_test "./data/snippets/snippets.LABEL" \
--word2idx "./data/snippets/word2idx.json" \
--word_vec "./data/snippets/$3_wv.npy" \
--alpha 0.1 \
--burn_in 1568
