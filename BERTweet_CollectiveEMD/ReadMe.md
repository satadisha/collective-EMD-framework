
Download data from shared drive (link in *data/ReadMe.md*)
Place model in *saved* folder before running notebook.

Pre-trained BERTweet model can be downloaded from Huggingface model hub.

Notebook finetunes pre-trained model on WNUT17 training file for EMD.

## Entity Classifiers

```
EntityClassifierI: syntactic + contextual embeddings, average pooling 
Model in entityClassifier/model_checkpoints
```

```
EntityClassifierII: only contextual embeddings, average pooling
Model in entityClassifier/model_checkpoints
```

```
EntityClassifierAlt: only contextual embeddings, learned pooling function
Model in entityClassifierAlt/model_checkpoints
```

