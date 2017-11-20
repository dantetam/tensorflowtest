Dante Tam
datam@berkeley.edu

A collection of machine learning experiments, primarily in the problem of converting human speech into computer text commands
i.e. this is the base for the conversational agent Stella. 

## IMPORTANT DIRECTORIES:

./tensorflowtest/word2vec
A condensed version of word2vec, the pretrained machine-learned vector representations of English words in dimension R^300.
These are limited to the approx. 50,000 most common words in English, and indexed by the first two letters, for easy indexing,
in either the web or offline application. 

## COMMANDS:

source activate tensorflow
Hopefully activate an environment with tensorflow, numpy, etc. installed for use 

python ./tensorflowtest/stella_softmax.py
Activate a TF experiment to classify commands by naive softmax regression (one-hot encoded vectors)


python ./tensorflowtest/twitter/twitter_lin_reg.py
See the following report (for the method used in this exact code, as well as better methods):
https://dantetam.github.io/src/twitter_ipynb.html

python ./tensorflowtest/cnn_text_git/train.py
python ./tensorflowtest/cnn_text_git/eval.py --eval_train --checkpoint_dir="./runs/1499926209/checkpoints/"
Train and evaluate a CNN text classification structure (citations in ./tensorflowtest/cnn_text_git/README.md)

python ./tensorflowtest/breastcancer/breast_cancer_softmax.py
Similar to previous softmax experiments, this tries to run a softmax logistic multiclass regression on the breast cancer dataset (Wisconsion, UCI archive).
This operates under the assumption that the data should not be overfitted and is not linearly separable. Other people's experience imply that the data 
is linearly separable, or at least separable in some reasonable polynomial kernel. 