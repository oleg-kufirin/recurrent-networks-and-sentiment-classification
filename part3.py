import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB

import re

# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.input_dim = 50
        self.hidden_dim = 128
        self.num_layers = 2
        self.linear1_dim = 64
        self.linear2_dim = 1
        self.dp = 0.2
        self.dropout = tnn.Dropout(self.dp)
        
        self.lstm = tnn.LSTM(input_size =self.input_dim, 
                             hidden_size=self.hidden_dim,
                             num_layers=self.num_layers,
                             batch_first=True,
                             bidirectional = True,
                             dropout=self.dp)

        self.fc1 = tnn.Linear(in_features=self.hidden_dim*2, 
                              out_features=self.linear1_dim)
        
        self.fc2 = tnn.Linear(in_features=self.linear1_dim, 
                              out_features=self.linear2_dim)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """        
        packed_input = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        packed_lstm_out, (hidden, cell) = self.lstm(packed_input)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        fc1_out = torch.nn.functional.relu(self.fc1(hidden))
        fc2_out = self.fc2(fc1_out) 

        out = torch.squeeze(fc2_out, 1)
        return out

class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch
    
    def tok(x):
        x = x.split()
        tokens = []
        for i in x:
          s = re.sub(r"<br", r"", i)
          s = re.sub(r"[^a-zA-z]+$", r"", s)
          s = re.sub(r"^[^a-zA-z]+", r"", s)
          if re.search("[^a-zA-z\']", s) == None:
            tokens.append(s)
          else:
            new_t = re.sub("[^a-zA-z0-9]", r" ", s)
            for k in new_t.split(): tokens.append(k)
        
        return tokens

    s_w = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 
                 'its', 'itself', 'they', 'them', 'their', 'theirs', 
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
                 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
                 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 
                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                 'through', 'during', 'before', 'after', 'above', 'below', 
                 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                 'under', 'again', 'further', 'then', 'once', 'here', 'there', 
                 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
                 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
                 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
                 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 
                 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 
                 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 
                 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 
                 'weren', 'won', 'wouldn', 'movie', 'film']
    
    text_field = data.Field(lower=True, include_lengths=True, tokenize=tok,
                            batch_first=True, 
                            preprocessing=pre, postprocessing=post,
                            stop_words = s_w)

def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return torch.nn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)
    print(len(textField.vocab.freqs))

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
#        net.eval()
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()
#        net.train()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
