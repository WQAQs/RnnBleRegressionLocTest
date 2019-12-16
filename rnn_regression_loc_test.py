import torch
from torch import nn
import numpy as np
import pandas as pd
import globalConfig


'''
每个timestamp输入采集到的一个ibeacon强度值
在 m 个timestamp后输出定位结果（x，y 坐标）
'''

Sample_Dataset_File = '.\\rnn_sample_set_4days.csv'
# set hyperameters
output_size = 2
n_epochs = 1000
hidden_dim = 12  # it's up to you
time_stamps = globalConfig.n_timestamps
n_layers = 10  # it's up to you
lr = 0.01

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers,
                          batch_first=True)  # batch_first – If True, then the input and output tensors are provided as
        # (batch, seq, feature). Default: False

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        # relu layer
        self.relu = nn.ReLU()

        # dropout layer
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        # hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        # rnn_out, hidden = self.rnn(x, hidden)
        # test = hidden[0]
        rnn_out, hidden = self.lstm(x)

        temp = rnn_out[:, -1, :]
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = temp.contiguous().view(-1, self.hidden_dim)

        # out = self.dropout(out)
        out = self.fc(out)
        out = self.relu(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We  78 8i 'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, 1, batch_size, self.hidden_dim).cuda()  #  !!!!!!!
        return hidden

def divide_sample_dataset(sample_dataset_file):
    sample_dataset = pd.read_csv(sample_dataset_file)
    train_dataset = sample_dataset.sample(frac=0.8,random_state=0)
    test_dataset = sample_dataset.drop(train_dataset.index)
    return train_dataset, test_dataset

def load_dataset(dataset):
    reference_tag = dataset.values[:, 0]
    data_input = dataset.values[:,5:] #包括index=5
    # data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    coordinates = dataset.values[:,1:3] #包括index=1，不包括index=3
    return data_input, coordinates,reference_tag

def load_data(data_file):
    dataset = pd.read_csv(data_file)
    data_input , coordinates, reference_tag = load_dataset(dataset)
    return data_input,coordinates,reference_tag


cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
# train_dataset,test_dataset = divide_sample_dataset(Sample_Dataset_File)
# train_data_input, train_data_target,train_reference_tag = load_dataset(train_dataset)
# test_data_input, test_data_target,test_reference_tag = load_dataset(test_dataset)
train_data_input, train_data_target,train_reference_tag = load_data(Sample_Dataset_File)

n_ibeacons = train_data_input.shape[1]
time_stamps = n_ibeacons
n = train_data_input.shape[0]
input_seqs = train_data_input.reshape(n, time_stamps, 1)
target_seqs = train_data_target


input_seqs = torch.from_numpy(input_seqs)
target_seqs = torch.Tensor(target_seqs)
input_seqs,target_seqs = input_seqs.type(torch.float32),target_seqs.type(torch.float32)


# get hyperparameters' value
input_size = 1


# Instantiate the model with hyperparameters
model = Model(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(cuda0)
criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Training Run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    input_seqs=input_seqs.cuda()
    output, hidden = model(input_seqs)
    # target_res = target_seqs.view(-1).long()
    target_seqs = target_seqs.cuda()
    loss = criterion(output, target_seqs)
    loss.backward()  # Does backpropagation and calculates gradients
    optimizer.step()  # Updates the weights accordingly

    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))


