**[This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

## Stella

This code was modified to solve a multi-class classification problem. Stella is a conversational agent that is given input text and returns the appropriate action.
Inspired by UC Berkeley's CS 170 and CS 189 courses in algorithms and machine learning, as well as natural language processing.

## References for Stella

Princeton University "About WordNet." WordNet. Princeton University. 2010. <http://wordnet.princeton.edu>

Mihalcea, Tarau. "TextRank: Bringing Order into Texts." University of North Texas. 2005. <https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf>

S. Brin and L. Page. 1998. The anatomy of a large-scale hyper-textual Web search engine. Computer Networks and ISDN Systems, 30(1–7).

Pak, Paroubek. Twitter as a Corpus for Sentiment Analysis and Opinion Mining. Universit ́e de Paris-Sud, Laboratoire LIMSI-CNRS. 2011. < http://web.archive.org/web/20111119181304/http://deepthoughtinc.com/wp-content/uploads/2011/01/Twitter-as-a-Corpus-for-Sentiment-Analysis-and-Opinion-Mining.pdf >

Tan, Steinbach, Kumar. "Association Analysis: Basic Concepts and Algorithms." Introduction to Data Mining. Pearson, 2005. < https://www-users.cs.umn.edu/~kumar/dmbook/ch6.pdf >

Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14, no. 3, pp 130-137.

Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. 2014. The Stanford CoreNLP Natural Language Processing Toolkit In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 55-60.

Manning, Christopher D, et al. An Introduction to Information Retrieval. Cambridge, England, Cambridge University Press, 2009.
Olah, Christopher. "Understanding LSTM Networks." Colah's Blog. N.p., 27 Aug. 2015. Web. 16 May 2017. .O. Vinyals, Q.V. Le. A Neural Conversational Model. ICML Deep Learning Workshop, arXiv 2015.

## Implementation

It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```bash
./train.py
```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## Original References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
