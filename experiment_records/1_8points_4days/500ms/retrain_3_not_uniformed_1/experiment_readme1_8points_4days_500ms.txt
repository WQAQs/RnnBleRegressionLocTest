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

retrain_pre_model: .\experiment_records\1_8points_4days\500ms\not_uniformed_1\model_1_8points_4days_500ms.pth
current_model_file : model1_1_8points_4days_500ms__01_15_105042.pth
evaluate dataset : valid_dataset1_8points_4days_500ms.csv
model valid Loss = 0.2603459358215332

