# Implementing Word2Vec

## Create environment

```
conda env create -f env.yml
conda activate a2
```

## Module test

```
python word2vec.py m
where m = sigmoid/ naiveSoftmaxLossAndGradient/ negSamplingLossAndGradient/ skipgram
```

## Full test

```
python word2vec.py

python sgd.py
```

## Download Stanford Sentiment Treebank (SST) dataset

```
sh get datasets.sh
```

## Train word vectors using SST dataset

```
python run.py
```

## Results

iter 40000: 9.812206

![Word2Vec](a2/word_vectors.png)

## Refer to ![handout](a2/handout2.pdf) for more detailed instructions