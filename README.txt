
In this repository we make our code available for the EQuANt model that we developed.
It is an extension of the QANet algorithm (https://arxiv.org/abs/1804.09541) that we extended with an answerability prediction module. This module allows our algorithm to perform well on the SQuAD 2 dataset that contains a number of non answerable questions that the QANet algorithm cannot cope with.

The paper describing our model and its results is available here: https://arxiv.org/abs/1907.00708


To start easily with the code:
- First, have a look at the prepro.py file to understand how the dataset is preprocessed in order to be used by the network.
- Then, have a look at the main.py file where all the specifications of the model are defined.
- The Model class in model.py is where the tensorflow implementation of the EQuANet is. It uses methods defined in layers.py.
- The methods.py file contains the train(), test(), and evaluate_batch() methods that allow to handle the model.

If you have any further questions, be sure to contact us.
