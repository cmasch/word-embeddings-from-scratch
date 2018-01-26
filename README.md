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

## References
[1] [Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec)<br>
[2] [Embeddings](https://www.tensorflow.org/programmers_guide/embedding)

## Author
Christopher Masch