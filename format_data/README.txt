模型说明：
mac的onehot编码+rssi值 输入RNN，输出预测坐标（x，y）
只使用部署的AP对应的mac

实验结果：
样本集（拆分作为训练，验证，测试集）                  MSELoss
rnn_test2//rnn_sample_set_onehot_test2.csv         21.2658