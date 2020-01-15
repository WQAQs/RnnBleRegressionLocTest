train_model_hyperparameter :
n_epochs = 100
hidden_dim = 12
batch_size = 256
n_layers = 1
lr = 0.01
earlystop_patience = 5

wordvec_uniform_flag = False

train_model_file : train_dataset1_8points_4days_500ms.csv
valid_model_file : valid_dataset1_8points_4days_500ms.csv

model_file : model_1_8points_4days_500ms.pth
evaluate dataset : valid_dataset1_8points_4days_500ms.csv
model valid Loss = 0.29623720049858093

