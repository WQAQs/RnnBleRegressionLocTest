train_model_hyperparameter :
n_epochs = 100
hidden_dim = 12
batch_size = 256
n_layers = 1
lr = 0.005
earlystop_patience = 5

wordvec_uniform_flag = True

train_model_file : train_dataset1_8points_4days_500ms.csv
valid_model_file : valid_dataset1_8points_4days_500ms.csv

retrain_pre_model: model1_1_8points_4days_500ms_01_15_101956.pth
current_model_file : model1_1_8points_4days_500ms__01_15_104654.pth
evaluate dataset : valid_dataset1_8points_4days_500ms.csv
model valid Loss = 4.103750705718994

