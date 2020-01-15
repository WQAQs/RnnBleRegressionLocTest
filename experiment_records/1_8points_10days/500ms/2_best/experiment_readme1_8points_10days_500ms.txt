train_model_hyperparameter :
n_epochs = 100
hidden_dim = 12
batch_size = 256
n_layers = 1
lr = 0.005
earlystop_patience = 5

wordvec_uniform_flag = True

train_model_file : train_dataset1_8points_10days_500ms.csv
valid_model_file : valid_dataset1_8points_10days_500ms.csv

retrain_pre_model: .\experiment_records\1_8points_4days\500ms\retrain_3_not_uniformed_1\model1_1_8points_4days_500ms_01_15_105042.pth
current_model_file : model1_1_8points_10days_500ms_01_15_114509.pth
evaluate dataset : valid_dataset1_8points_10days_500ms.csv
model valid Loss = 0.21111100912094116

