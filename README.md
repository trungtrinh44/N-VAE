# Nested Variational Autoencoder for Topic Modeling with Word Vectors

This is the implementation of the **Nested Variational Autoencoder**, which is a neural network that perform topic modeling and can leverage word embeddings. The experiment datasets are hosted at [\[link\]](https://drive.google.com/drive/folders/1Cv2mWzOB_ulZdWfL3FPkK4sY8KEf9IOA?usp=sharing).

#### Dependencies
This implementation requires **Python >= 3.5**.
Package dependencies are included in **requirements.txt**

#### Usage
`$ python train.py --layers --save_path --ntopics --epoch [--beta1] [--beta2] [--batch_size] --lr --temp --train_data [--test_data] [--label_test] --word2idx [--word_vec] [--wv_size] [--train_wv] [--alpha] --burn_in [--accumulate_metrics] [--experiment_name]` 
where arguments in [] are optional
- `layers`: specify the size of the fully connected layers. For example: `128,128` means that there are 2 layers, each of them has 128 units.
- `save_path`: the path of the directory to save the training result.
- `ntopics`: number of topics
- `epoch`: number of epochs.
- `beta1` and `beta2`: the hyperparameters of the Adam Optimizer.
- `batch_size`: batch size.
- `lr`: the maximum learning rate and the number of iterations to reach the maximum value, separate by a comma. For example: `8e-3,10` means that the maximum learning rate is `8e-3` and this value is reached after `10` iterations.
- `temp`: the minimum value and the number of iterations to reach this value of temperature of the Gumbel-Softmax. For example: `0.5,20` means that the minimum temperature is `0.5` and this value is reached after `20` iterations.
- `train_data`: path to the training data where each example is a list of pairs of **(word, count)** (look at the experiment data for more detail).
- `test_data`: path to the test data with similar format to the training data.
- `label_test`: path to the labels of the test data for document clustering evaluation.
- `word2idx`: path to the `word2idx` file which is a `json` file consists of an object where its keys are the words in the vocabulary and its values are the indices. Index start from 0.
- `word_vec`: path to the word embedding files which is a 2-D numpy array where each row corresponding to a vector of a word whose index matches the row index. Row 0 is a vector of zeros.
- `wv_size`: if the path to the pretrained word embedding `word_vec` is not provided, this argument indicates the size of the randomly initialized word embeddings.
- `train_wv`: a boolean argument indicates whether or not to fine tune the word embeddings.
- `alpha`: the parameter of the Dirichlet prior distribution.
- `burn_in`: the number of iterations before training the `alpha` value.
- `accumulate_metrics`: the path to the file to save the experiment result if test data is provided.
- `experiment_name`: the name of the experiment to save in the `accumulate_metrics` file.

To run the experiment, download the experiment data and run the corresponding scripts. For example, to run the experiment on the `N20small` dataset with `6` topics and `GloVe` pretrained word embeddings :

`bash N20small.sh runs/N20small 6 glove N20small,6,glove runs/result.csv`

To run the model on a new dataset, please format the dataset according to the experiment data. Remember that word index start from 0.