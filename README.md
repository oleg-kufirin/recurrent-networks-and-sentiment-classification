# Recurrent Networks and Sentiment Classification
## Assignment for *"Neural Networks and Deep Learning"* class at UNSW
### Received mark: 21/24

This assignment is divided into three parts:
* Part 1 contains simple PyTorch questions focused on reccurent neural networks designed to get you started and familiar with this part of the library.
* Part 2 involves creating specific recurrent network structures in order to detect if a movie review is positive or negative in sentiment.
* Part 3 is an unrestricted task where marks will be assigned primarily on final accuracy, and you may implement any network structure you choose.

**Provided Files**

Copy the archive *hw2.zip* into your own filespace and unzip it. This should create an *hw2* directory with three skeleton files *part1.py*, *part2.py* and *part3.py* as well as two subdirectories: *data* and *.vector_cache*

Your task is to complete the skeleton files according to the specifications in this document, as well as in the comments in the files themselves. Each file contains functions or classes marked TODO: which correspond to the marking scheme shown below. This document contains general information for each task, with in-code comments supplying more detail. Parts 1 and 2 in this assignment are sufficiently specified to have only one correct answer (although there may be multiple ways to implement it). If you feel a requirement is not clear you may ask for additional information on the course forum.

There is also an additional file, *imdb_dataloader.py*. This is used to load the dataset provided to you in *./data* for parts 2 and 3. It will also be used in our testing. Do not modify this file.

**Part 1 [4 marks]**  
For Part 1 of the assignment, you should work through the file part1.py and complete the functions where specified.

**Part 2 [8 marks]**  
For Part 2, you will develop several models to solve a text classification task on movie review data. The goal is to train a classifier that can correctly identify whether a review is positive or negative. The labeled data is located in data/imdb/aclimdb and is split into train (training) and dev (development) sets, which contain 25000 and 6248 samples respectively. For each set, the balance between positive and negative reviews is equal, so you don't need to worry about class imbalances.

You should take at least 10 minutes to manually inspect the data so as to understand what is being classified. In the entire collection, no more than 30 reviews are allowed for any given movie because reviews for the same movie tend to have correlated ratings. Further, the train and dev sets contain a disjoint set of movies, so no significant performance is obtained by memorizing movie-unique terms and their association with observed labels. In the labeled train/dev sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included.

The provided file *part2.py* is what you need to complete. This code makes heavy use of *torchtext*, which aims to be the NLP equivelent to torchvision.

For all tasks in this part, if arguments are not specified assume PyTorch defaults.

**Task 1: LSTM Network**  
Implement an LSTM Network according to the function docstring. When combined with an appropriate loss function this model should achieve ~81% when run using the provided code.

**Task 2: CNN Network**  
Implement a CNN Network according to the function docstring. When combined with an appropriate loss function this model should achieve ~82% when run using the provided code.

**Task 3: Loss function**  
Define a loss function according to the function docstring.

**Task 4: Measures**  
Return (in the following order), the number of true positive classifications, true negatives, false positives and false negatives. True positives are positive reviews correctly identified as positive. True negatives are negative reviews correctly identified as negative. False positives are negative reviews incorrectly identified as positive. False negatives are postitive reviews incorrectly identified as negative.

**Part 3 [12 marks]**  
The goal of this section is to simply achieve the highest accuracy you can on a holdout test set (i.e. a section of the dataset that we do not make available to you, but will test your model against).

You may use any form of model and preprocessing you like to achieve this, provided you adhere to the constraints listed below.

The provided code *part3.py* is essentially the same as part2.py except that it reports the overall accuracy, and at the end of training it saves the model in a file called model.pth (which you will need to submit). A good starting point would be to copy the relevant sections of code from your best model for *part2.py* into *part3.py*.

Your code must be capable of handling various batch sizes. You can check this is working ok with the submission tests. The code provided in *part3.py* already does this.

You can modify and change the code however you would like, however you MUST ensure that we can load your code to test it. This is done in the following way:

* Import and create and instance of your network from the *part3.py* file you submit.
* Restore this network to its trained state using the *state-dict* you provide.
* Load a test dataset, preprocessing each sample using the *text_field* you specify in your *PreProcessing* class.
* Feed this dataset into your model and record the accuracy.

You should check the docs on the *torchtext.data.Field* class to understand what you can and canâ€™t do to the input.

Specific to preprocessing, you may add a post-processing function to the field, as long as that function is also declared in the Preprocessing class. You may also add a custom tokenizer, stopwords, etc. Note that none of this is necessarily required, but it is possible.

You may wish to carry out some data augmentation. This is because in practice more data will outperform a better model. Data augmentation (transforming the data you have been provided and creating a new sample with the same label) is allowed. You are allowed to modify the main() function to create additional data in place. You may not call any remote API's when doing this. Assume the test environment has no internet connection.

You may NOT download or load data other than what we have provided. If we find your submitted model has been trained on external data you will receive a mark of 0 for the assignment.

Marks for part 3 will be based primarily on the accuracy your model achieves on the unseen test set.

When you submit part 3, in addition to the standard checks that we can run and evaluate your model, you will also see an accuracy value. This is the result of running your model on a very small number of held-out training examples (~600). These samples can be considered representative of the final test set, however the final accuracy will be calculated from significantly more samples (~18 000). The submission test should take no longer than 10s to run.

**Constraints**
* Saved model state-dict must be under 5MB and you cannot load external assets in the network class
* Model must be defined in a class named network.
* The save file you submit must be generated by the *part3.py* file you submit.
* Must use 50d GloVe Vectors for vectorization. This means no pretraining. We are solely interested in the problem as a classification task, so trying to use something like BERT or GPT-2 is not relevant.
* While you may train on a GPU, you must ensure your model is able to be evaluated (i.e. perform inference) on a CPU.
