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
--lr "8e-3,10" \
--temp "0.7,10" \
--train_data "./data/Twitter/Twitter.npy" \
--test_data "./data/Twitter/Twitter.npy" \
--label_test "./data/Twitter/Twitter.LABEL" \
--word2idx "./data/Twitter/word2idx.json" \
--word_vec "./data/Twitter/$3_wv.npy" \
--alpha 0.1 \
--burn_in 640
