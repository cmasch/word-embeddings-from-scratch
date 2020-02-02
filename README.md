### *** Not working with Tensorflow 2.x ***
I will update the code asap.


# Word embeddings from scratch and visualization
If you are working with documents one approach is to create word embeddings that allows to represent words with similar meaning.

In this [jupyter notebook](https://github.com/cmasch/word-embeddings-from-scratch/blob/master/Create_Embeddings.ipynb) I would like to show how you can create embeddings from scratch using `gensim` and visualize them on `TensorBoard` in a simple way.<br>
Some time ago I tried the build-in method [word2vec2tensor](https://radimrehurek.com/gensim/scripts/word2vec2tensor.html) of `gensim` to use `TensorBoard`, but without success. Therefore I implemented this version in combination with `TensorFlow`.

For this example I used a subset of 200000 documents of the [Yelp dataset](https://www.yelp.com/dataset). This is a great dataset that included different languages but mostly english reviews.<br>

As you can see in my animation, it learns the representation of similiar words from scratch. German and other languages are also included!<br>
<img src="./embedding.gif"><br>
You can improve the results by tuning some parameters of word2vec, using t-SNE or modifying the preprocessing.

## Usage
Because of the [dataset license](https://s3-media2.fl.yelpcdn.com/assets/srv0/engineering_pages/e926cc12796d/assets/vendor/yelp-dataset-license.pdf) I can't publish my training data nor the trained embeddings. Feel free to use the notebook for your own dataset or request the data on [Yelp](https://www.yelp.com/dataset).
Just put your text-files in the defined directory `TEXT_DIR`. Everything will be saved in folder defined by `MODEL_PATH`.

Finally start TensorBoard:
```
tensorboard --logdir emb_yelp/
```

## Using trained embeddings in Keras
If you would like to use your own trained embeddings for neural networks, you can load the trained weights (vectors) in an [embedding layer](https://keras.io/layers/embeddings/) (e.g. Keras). This is really useful, especially if you have just a few samples to train your network on. Another reason is that existing pre-trained models like Google word2vec or GloVe are maybe not sufficient because they are not task-specific embeddings.

If you need an example how to use trained embeddings of gensim in Keras, please take a look at the code snippet below. This is similiar to this [jupyter notebook](https://github.com/cmasch/cnn-text-classification/blob/master/Evaluation.ipynb) where I used GloVe. But loading gensim weights is quite a bit different.

```python
def get_embedding_weights(gensim_model, tokenizer, max_num_words, embedding_dim):
    model = gensim.models.Word2Vec.load(gensim_model)
    embedding_matrix = np.zeros((max_num_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in model.wv.vocab and i < max_num_words:
            embedding_vector = model.wv.vectors[model.wv.vocab[word].index]
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
    

emb_weights = get_embedding_weights(gensim_model='emb_yelp/word2vec',
                                    tokenizer=tokenizer,
                                    max_num_words=MAX_NUM_WORDS,
                                    embedding_dim=EMBEDDING_DIM
                                   )

embedding_layer = Embedding(input_dim=MAX_NUM_WORDS,
                            output_dim=EMBEDDING_DIM,
                            input_length=MAX_SEQ_LENGTH,
                            weights=[emb_weights],
                            trainable=False
                           )
```

## References
[1] [Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec)<br>
[2] [Embeddings](https://www.tensorflow.org/programmers_guide/embedding)

## Author
Christopher Masch
