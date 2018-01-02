# Master thesis: Integration of Image and Word Embeddings for Descriptive Image Similarity
### David Gustafsson \& Tobias Lindberg
Repository for the master thesis "Integration of Image and Word Embeddings for Descriptive Image Similarity" using a convolutional neural network.

#### To run:
Make sure you're in the project root dir:
```sh
$ pwd
/Users/tobias/Desktop/exjobb/git_master_thesis
```

Specify your model in the .py-file and run it as:
```sh
python3 ./code/slim_inception_v3.py
```

To follow you model's training progress:
```sh
tensorboard --logdir=runs
```

To run the beautiful demo:
```sh
python3 ./code/demo.py
```
