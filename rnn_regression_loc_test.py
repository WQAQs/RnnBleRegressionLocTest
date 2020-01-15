import torch
from torch import nn
import numpy as np
import pandas as pd
import globalConfig
import ast
import getWordVector
import math
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping
import torch.utils.data as Data
from sklearn.utils import shuffle
import os.path
import time

'''
每个timestamp输入采集到的一个ibeacon强度值
在 m 个timestamp后输出定位结果（x，y 坐标）
'''

 # 文件目录
Sample_Dataset_File = globalConfig.sample_dataset_file  # 模型使用的样本集文件
train_dataset_file = globalConfig.train_dataset_file
valid_dataset_file = globalConfig.valid_dataset_file
test_dataset_file = globalConfig.test_datset_file
pre_network_model_file = globalConfig.pre_network_model_file
original_saved_model_file = globalConfig.save_network_model_file
current_saved_model_file = ""
evaluate_model_file = globalConfig.evaluate_model_file
error_distance_file = globalConfig.error_distance_file

output_size = 2  # 模型输出一个样本的结果的size, 对应（x，y 坐标）

# 模型超参数设置
n_epochs = globalConfig.n_epochs
hidden_dim = globalConfig.hidden_dim
batch_size = globalConfig.batch_size
n_layers = globalConfig.n_layers
lr = globalConfig.lr
earlystop_patience = globalConfig.earlystop_patience

# 词向量
vocab_size = len(getWordVector.wordvec)
embedding_dim = getWordVector.n_valid_ibeacon + 1
word_dim = embedding_dim
# divide_dataset_flag = False
# best_train_pred = []

max_ibeacon_len = 0  # 一个样本中包含的最多的ibeacon个数
min_loss = 1000
# time_stamps = globalConfig.n_timestamps

retrain_model_flag = globalConfig.retrain_model_flag

class Model(nn.Module):
    def __init__(self, embedding_dim, input_size, output_size, hidden_dim, n_layers, wordvec_uniform_flag=True):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # embedding
        # 用采集的数据自己建立词向量表
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        if wordvec_uniform_flag:
            pretrained_embeddings = np.array(getWordVector.wordvec_uniformed)
        else:
            pretrained_embeddings = np.array(getWordVector.wordvec)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        # 使用pytorch提供的embedding模块
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.embed.weight.requires_grad = False

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

    def forward(self, sentence):
        x = self.embed(sentence)
        # x = self.embeddings(sentence)
        rnn_out, hidden = self.lstm(x)
        temp = rnn_out[:, -1, :]
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = temp.contiguous().view(-1, self.hidden_dim)
        # out = self.dropout(out)
        out1 = self.fc(out)
        out = self.relu(out1)
        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We  78 8i 'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, 1, batch_size, self.hidden_dim).cuda()  #  !!!!!!!
        return hidden

# def divide_sample_dataset(sample_dataset):
#     train_dataset = sample_dataset.sample(frac=0.8, random_state=0)
#     valid_dataset = sample_dataset.drop(train_dataset.index)
#     train_dataset.to_csv(train_dataset_file, index=False, encoding='utf-8')
#     valid_dataset.to_csv(valid_dataset_file, index=False, encoding='utf-8')


def load_dataset(dataset):
    reference_tag = dataset.values[:, 0]
    data_input = dataset.values[:, 5] #包括index=5
    # data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    coordinates = dataset.values[:, 1:3] #包括index=1，不包括index=3
    return data_input, coordinates, reference_tag


def my_load_data(train_dataset_file, valid_dataset_file):
    train_dataset = pd.read_csv(train_dataset_file)
    valid_dataset = pd.read_csv(valid_dataset_file)
    train_input, train_coordinates, train_reference_tag = load_dataset(train_dataset)
    valid_input, valid_coordinates, valid_reference_tag = load_dataset(valid_dataset)
    train_coordinates = train_coordinates.astype(float)
    valid_coordinates = valid_coordinates.astype(float)
    return train_input, train_coordinates, train_reference_tag, valid_input, valid_coordinates, valid_reference_tag


def load_data(dataset_file):
    dataset = pd.read_csv(dataset_file)
    data_input, data_out, data_reference_tag = load_dataset(dataset)
    data_out = data_out.astype(float)
    return data_input, data_out, data_reference_tag


def load_train_valid_data(train_dataset_file, valid_dataset_file):
    train_input, train_coordinates, train_reference_tag = load_data(train_dataset_file)
    valid_input, valid_coordinates, valid_reference_tag = load_data(valid_dataset_file)
    return train_input, train_coordinates, train_reference_tag, valid_input, valid_coordinates, valid_reference_tag


def csvstr2list(data):
    new_data = []
    for i in data:
        temp = ast.literal_eval(i)
        new_data.append(temp)
    return new_data

# def getmask_padding(list, max_len):
#     for item in list:
#         n_padding = max_len - len(item)
#         for i in range(n_padding):
#             item.append(0)

def padding(list, max_len):
    for item in list:
        n_padding = max_len - len(item)
        for i in range(n_padding):
            item.append(0)


def get_max_ibeacons_n(list):
    max_len = 0
    for item in list:
        l = len(item)
        if max_len < l:
            max_len = l
    return max_len


def calculate_distance(result):
    pred_coordinatesx,pred_coordinatesy = result[0], result[1]
    true_coordinatesx,true_coordinatesy = result[2], result[3]
    error_x2, error_y2 = math.pow(pred_coordinatesx - true_coordinatesx, 2), \
                         math.pow(pred_coordinatesy - true_coordinatesy, 2)
    error_distance = math.sqrt(error_x2 + error_y2)  # 求平方根
    return error_distance


def strim_data2batches(input, coordinates,reference_tag):
    l = len(input)
    n = int(l / batch_size)
    i = n * batch_size
    del input[i:]   # 这里input是list
    coordinates = np.delete(coordinates, [n for n in range(i, l)], axis=0)   # 这里coordinates是数组只能这么处理
    reference_tag = np.delete(reference_tag, [n for n in range(i, l)], axis=0)
    return input, coordinates, reference_tag


def process_data(input, coordinates, reference_tag):
    input = csvstr2list(input)  # 存储在csv中的数组，在读取后变成了str类型：如 '[1782, 2004, 2841, 4101, 1781, 2001, 3055,
                                # 3889, 2841, 1781, 1998, 3888, 2839, 1998, 3055, 3888, 2841, 1782, 2422, 1997, 2842, 2639]'
                                # 要解析还原一下
    max_ibeacon_len = get_max_ibeacons_n(input)
    padding(input, max_ibeacon_len)
    input, coordinates, reference_tag = strim_data2batches(input, coordinates, reference_tag)
    input = np.array(input)
    # input = input.reshape(-1, batch_size,max_ibeacon_len)
    new_input = torch.Tensor(input).long()
    new_coordinates = torch.Tensor(coordinates)
    new_input = new_input.cuda()
    new_coordinates = new_coordinates.cuda()
    return new_input, new_coordinates, reference_tag


# Train the Model using Early Stopping
def train_model(model, criterion, optimizer, train_loader, valid_loader):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(earlystop_patience, verbose=True)
    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (data, target) in enumerate(train_loader):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output, _ = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            output = output[0]
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    avg_train_losses = get_average(avg_train_losses)
    avg_valid_losses = get_average(avg_valid_losses)
    return model, avg_train_losses, avg_valid_losses

def get_average(list):
    sum = 0
    for item in list:
        sum += item
    return sum/len(list)

# def train_mymodel(model, x_train, y_train, x_valid, y_valid):
#     # Training Run
#     for epoch in range(1, n_epochs + 1):
#         optimizer.zero_grad()  # Clears existing gradients from previous epoch
#         input_seqs = input_seqs.cuda()
#         # input_seqs = input_seqs
#         output, hidden = model(input_seqs)
#         # target_res = target_seqs.view(-1).long()
#         target_seqs = target_seqs.cuda()
#         # target_seqs = target_seqs
#         loss = criterion(output, target_seqs)
#         loss.backward()  # Does backpropagation and calculates gradients
#         optimizer.step()  # Updates the weights accordingly
#         if epoch % 10 == 0:
#             print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
#             print("Loss: {:.4f}".format(loss.item()))
#     return


def clear_dir(root_path):
    for i in os.listdir(root_path):
        path_file = os.path.join(root_path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)


def my_evaluate_model(model_file, dataset_file):
    clear_dir(globalConfig.experiment_running_results_dir)  # 清空experiment_running_results_dir文件夹的内容
    data_input, data_target, reference_tag, batch_loader = initialize_data(dataset_file)  # 准备评估模型用的数据

    current_model = torch.load(model_file)  # 加载待评估的模型
    pred_result, _ = current_model(data_input)  # get the prediction results
    retrain_msg = ""
    if retrain_model_flag:
        retrain_msg = "retrain_pre_model: " + globalConfig.pre_network_model_file + "\n"
    # 保存实验说明文件experiment_readme.txt
    loss = criterion(pred_result, data_target)
    loss_msg = "model valid Loss = " + str(loss.item()) + "\n"
    dataset_file_name = dataset_file.split("\\")[-1]  # 获取不带路径的文件名
    evaluate_dataset_msg = "evaluate dataset : " + dataset_file_name + "\n"
    model_file_name = model_file  # 获取不带路径的文件名
    evaluate_model_msg = "current_model_file : " + model_file_name + "\n"
    train_dataset_file_name = "train_model_file : " + train_dataset_file.split("\\")[-1] + "\n"
    valid_dataset_file_name = "valid_model_file : " + valid_dataset_file.split("\\")[-1] + "\n"
    wordvec_uniform_flag_msg = "wordvec_uniform_flag = " + str(globalConfig.wordvec_uniform_flag) + "\n"
    msg = globalConfig.train_model_hyperparameter_msg + "\n" + wordvec_uniform_flag_msg + "\n" \
          + train_dataset_file_name + valid_dataset_file_name + "\n" + retrain_msg + evaluate_model_msg \
          + evaluate_dataset_msg + loss_msg + "\n"
    print(msg)
    text_create(msg)

    # 数据处理
    pred_result = pred_result.cpu().tolist()
    data_target = data_target.cpu().tolist()
    pred_result = np.array(pred_result)
    data_target = np.array(data_target)
    results = np.hstack((pred_result, data_target))
    results_df = pd.DataFrame(results, columns=['pred_coordinates_x', 'pred_coordinates_y', 'true_coordinates_x', 'true_coordinates_y'])

    error_distance = list(map(calculate_distance, results))
    error_distance = np.array(error_distance)
    error_distance_bypoint = np.hstack((error_distance.reshape(-1, 1), reference_tag.reshape(-1, 1)))
    error_distance_bypoint_df = pd.DataFrame(error_distance_bypoint, columns=['error_distance', 'point_reference_tag'])

    evaluate_df = pd.concat([results_df, error_distance_bypoint_df], axis=1)
    evaluate_df.to_csv(globalConfig.evaluate_prediction_file)   # 保存evaluate_df到csv文件

    error_over1m_df = error_distance_bypoint_df[error_distance_bypoint_df['error_distance'] > 1.0]
    error_over1m_df = error_over1m_df.sort_values(by='error_distance')
    error_mean = error_over1m_df['error_distance'].mean()
    error_over1m_df.to_csv(error_distance_file, index=False, encoding='utf-8')   # 保存error_over1m_df到csv文件

    # 绘制error_distribution图
    plt.figure()
    error_len = len(error_distance)
    plt.scatter(list(range(0, error_len)), error_distance, s=6)
    plt.title("Prediction Distance Error By Point")
    plt.ylabel("Prediction Distance Error(/m)")
    plt.savefig(globalConfig.savefig_error_distribution_file)
    # plt.show()

    groupby_df = evaluate_df.groupby(['point_reference_tag'])
    true_coordinates_x, true_coordinates_y = evaluate_df['true_coordinates_x'].values.tolist(), evaluate_df['true_coordinates_y'].values.tolist()

    # 绘制coordinates_distribution图
    plt.figure()
    for point_reference_tag, group_data in groupby_df:
        pred_coordinates_x, pred_coordinates_y = group_data['pred_coordinates_x'].values.tolist(), group_data['pred_coordinates_y'].values.tolist()
        plt.scatter(pred_coordinates_x, pred_coordinates_y, s=6)
    plt.scatter(true_coordinates_x, true_coordinates_y, s=18, marker='p')
    plt.xlabel('coordinate_x(/m)')
    plt.ylabel('coordinate_y(/m)')
    plt.axis('equal')
    plt.axis('square')
    plt.savefig(globalConfig.savefig_coordinates_distribution_file)
    # plt.show()  # 在pycharm中显示绘图的窗口

    # 绘制error cdf图
    cdf(error_distance)


def cdf(data):
    # pd.DataFrame(data).to_csv("erro_cdf_dist.csv", header=None, index=False)
    hist, bins = np.histogram(data, 100)
    bins = bins[1:]
    for i in range(1, len(hist)):
        hist[i] = hist[i]+hist[i-1]
    hist = hist/len(data)
    plt.figure()
    plt.title("Prediction Distance Error CDF")
    plt.xlabel('Prediction Distance Error (/m)')
    plt.ylabel('CDF')
    plt.plot(bins, hist)
    plt.savefig(globalConfig.savefig_error_cdf_file)
    # plt.show()



# 创建一个txt文件，文件名为mytxtfile,并向文件写入msg
def text_create(msg):
    full_path = globalConfig.experiment_readme_file  # 也可以创建一个.doc的word文档
    # if os.path.exists(full_path):
    file = open(full_path, 'w')
    file.write(msg)  # msg也就是下面的Hello world!
    file.close()


def generate_data_loader(x_tensor, y_tensor):
    torch_dataset = Data.TensorDataset(x_tensor, y_tensor)
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=batch_size,
                             shuffle=True)
    return loader


def run_experiment(repeat=10):
    global evaluate_model_file
    global current_saved_model_file
    global min_loss
    global best_train_pred
    global criterion

    for i in range(repeat):
        print("\n" + "repeat = " + str(i) + "\n")
        if retrain_model_flag:
            model = torch.load(globalConfig.pre_network_model_file)  # 加载需要再训练的模型
            criterion = nn.MSELoss()  # 设置损失函数类型
            # criterion = nn.L1Loss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 设置优化器类型
        else:
            # Instantiate the model with hyperparameters
            model = Model(embedding_dim=embedding_dim, input_size=word_dim
                          , output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers,
                          wordvec_uniform_flag=globalConfig.wordvec_uniform_flag)  # 初始化一个新的model
            model.to(cuda0)  # 将模型部署在gpu上
            criterion = nn.MSELoss()  # 设置损失函数类型
            # criterion = nn.L1Loss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 设置优化器类型
        my_model, avg_train_losses, avg_valid_losses = train_model(model, criterion,
                                                                   optimizer, train_batch_loader, valid_batch_loader)
        if avg_valid_losses < min_loss:
            min_loss = avg_valid_losses
            best_model = my_model
    current_saved_model_file = original_saved_model_file + "_" \
                               + time.strftime("%m_%d_%H%M%S", time.localtime()) + ".pth"
    torch.save(best_model, current_saved_model_file)
    evaluate_model_file = current_saved_model_file


def initialize_data(data_file):   # 初始化数据：加载并且预处理数据,生成数据加载器
    data_input, data_coordinates, data_reference_tag = load_data(data_file)
    data_input, data_coordinates, data_reference_tag = process_data(data_input, data_coordinates, data_reference_tag)
    data_batch_loader = generate_data_loader(data_input, data_coordinates)  # 生成batch数据加载器
    return data_input, data_coordinates, data_reference_tag, data_batch_loader


if __name__ == '__main__':
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 判断cuda是否可用
    criterion = nn.MSELoss()  # 设置损失函数类型

    train_input, train_coordinates, \
    train_reference_tag, train_batch_loader = initialize_data(train_dataset_file)  # 加载和处理训练集数据
    valid_input, valid_coordinates, \
    valid_reference_tag, valid_batch_loader = initialize_data(valid_dataset_file)  # 加载和预处理验证集数据

    # run_experiment(globalConfig.model_repeat)  # 训练模型
    valid_dataset_file = globalConfig.sample_dataset_file
    my_evaluate_model(evaluate_model_file, valid_dataset_file)  # 评估模型


