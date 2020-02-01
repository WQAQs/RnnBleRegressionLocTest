import os

##########################
# getTrainSample.py 设置 #
##########################

####### 数据预处理参数设置  #######
timeInterval = 500  # 一个样本的时间，单位ms，即定位的时间间隔
####### 文件目录设置  ########
root_data_dir = ".\\data"
point_range = "1_8points"
n_days = "7days"
time_interval = str(timeInterval) + "ms"
resource_data_dir = root_data_dir + "\\resource_data"
all_raw_txt_data_dir = resource_data_dir + "\\all_raw_txt_data"
all_unlabeled_csv_dir = resource_data_dir + "\\all_unlabeled_csv"
# all_labeled_csv_dir = resource_data_dir + "\\all_labeled_csv"
root_txt_dir = resource_data_dir + "\\all_raw_txt_data\\train"  # 原始数据存在的文件夹
root_csv_dir = resource_data_dir + "\\points_csv\\train"  # 转换的csv文件目标文件夹
ibeaconFilePath = resource_data_dir + "\\ibeacon_mac_count_day1217.csv"  # ibeacon 统计文件
##### 生成样本集使用的数据来源 #######
# all_labeled_csv_root_dir = root_data_dir + ".\\labeled_csv_data"
generate_sampleset_all_labeled_csv_dir = resource_data_dir + "\\generate_sampleset_all_labeled_csv"  # 用来做样本集的数据文件夹
all_labeled_csv_dir = resource_data_dir + ".\\all_labeled_csv"  # 用来做样本集的数据文件夹
reference_points_coordinates_file = resource_data_dir + "\\sacura_reference_points_coordinates.csv"  # 参考点坐标文件
valid_ibeacon_file = resource_data_dir + "\\valid_ibeacon_mac.csv"  # 部署的有效ibeacon文件
##### 生成的样本集保存的位置 ######
sampleset_dir = root_data_dir + "\\sampleset_data"
sample_dataset_file = sampleset_dir + "\\onehot_sampleset" + point_range + "_" + n_days + "_" + time_interval + ".csv"  # 保存样本数据集的文件
train_dataset_file = sampleset_dir + "\\train_dataset" + point_range + "_" + n_days + "_" + time_interval + ".csv"  # 保存训练集的文件
valid_dataset_file = sampleset_dir + "\\valid_dataset" + point_range + "_" + n_days + "_" + time_interval + ".csv"  # 保存验证集的文件
# test_datset_file = testset_dir + "\\1_8points_7days_500ms\\onehot_sampleset1_8points_7days_500ms.csv"  # 保存测试集的文件


##################################
# rnn_regression_loc_test.py 设置#
##################################
model_repeat = 10  # 每一次run_experiment中训练出几个model
######## model的设置 #########
n_epochs = 100
hidden_dim = 12
batch_size = 2048*16
n_layers = 1
lr = 0.005
earlystop_patience = 5
wordvec_uniform_flag = True
###### 再训练模型的标志 ######
retrain_model_flag = True


train_model_hyperparameter_msg = "train_model_hyperparameter :" + "\n" \
                                 + "n_epochs = " + str(n_epochs) + "\n" \
                                 + "hidden_dim = " + str(hidden_dim) + "\n" \
                                 + "batch_size = " + str(batch_size) + "\n" \
                                 + "n_layers = " + str(n_layers) + "\n" \
                                 + "lr = " + str(lr) + "\n" \
                                 + "earlystop_patience = " + str(earlystop_patience) + "\n"
# n_timestamps = 10  # 模型超参数设置 目前没用到
# wordvec_file = ".\\wordvec_file.csv"  # 存储词向量的文件，目前没用到
####### 训练 验证 测试 model 使用的数据 #######
trainset_dir = sampleset_dir + "\\train"
testset_dir = sampleset_dir + "\\test"
model_sample_dataset_file = trainset_dir + "\\1_8points_33days_500ms\\onehot_sampleset1_8points_33days_500ms.csv"
model_train_dataset_file = trainset_dir + "\\1_8points_33days_500ms\\train_dataset1_8points_33days_500ms.csv"  # model训练集
model_valid_dataset_file = trainset_dir + "\\1_8points_33days_500ms\\valid_dataset1_8points_33days_500ms.csv"  # model验证集
model_test_dataset_file = testset_dir + "\\1_8points_7days_500ms\\onehot_sampleset1_8points_7days_500ms.csv"  # model测试集
###### 保存实验结果的目录设置  ######
experiment_running_results_dir = ".\\experiment_running_results"
experiment_records_dir = ".\\experiment_records"
# pre_network_model_file = experiment_records_dir \
#                          + "\\1_8points_4days\\500ms" \
#                            "\\retrain_3_not_uniformed_1\\model1_1_8points_4days_500ms_01_15_105042.pth"  # 之前训练出的网络模型文件
pre_network_model_file = experiment_records_dir + "\\best_model\\model1_1_8points_4days_500ms_01_15_105042.pth"

evaluate_model_file = experiment_records_dir + "\\1_8points_10days\\500ms\\2_best" + \
                      "\\model1_1_8points_10days_500ms_01_15_114509.pth"
save_network_model_file = experiment_running_results_dir + "\\model1_" \
                          + point_range + "_" + n_days + "_" + time_interval  # 保存网络模型的文件
error_distance_file = experiment_running_results_dir + "\\error_distance_over1m" \
                      + point_range + "_" + n_days + "_" + time_interval + ".csv"  # 统计模型预测误差距离超过1m的数据文件
evaluate_prediction_file = experiment_running_results_dir + "\\evaluate_prediction" \
                           + point_range + "_" + n_days + "_" + time_interval + ".csv"  # 保存模型预测结果的文件
savefig_coordinates_distribution_file = experiment_running_results_dir + "\\coordinates_distribution" \
                                        + point_range + "_" + n_days + "_" + time_interval + ".jpg"
savefig_error_distribution_file = experiment_running_results_dir + "\\error_distribution" \
                                  + point_range + "_" + n_days + "_" + time_interval + ".jpg"
savefig_error_cdf_file = experiment_running_results_dir + "\\error_cdf" \
                                  + point_range + "_" + n_days + "_" + time_interval + ".jpg"
experiment_readme_file = experiment_running_results_dir + "\\experiment_readme" \
                         + point_range + "_" + n_days + "_" + time_interval + ".txt"



######################
# 重设置根目录 #
######################
'''设置工作默认目录，使数据保持统一的存放格式，方便管理，若路径不存在将创建相应的目录及其结构'''
# #### 实验根目录设置 #####
# data_name = "rnn_test2"
# def copy_dirs(source, dist):
#     """
#     复制source下所有目录到dist目录中，若dist不存在将创建
#     :param source: string
#         源目录
#     :param dist: string
#         目标目录
#     :return: nothing
#     """
#     if not os.path.exists(dist):
#         os.makedirs(dist)
#         children = os.listdir(source)
#     for child_dir in children:
#         next_source = os.path.join(source, child_dir)
#         if os.path.isdir(next_source):
#             next_dist = os.path.join(dist, child_dir)
#             copy_dirs(next_source, next_dist)

# stableCount = 40  # 维度设置，暂时改为直接确定选取的mac数目，目前没用到
# base_dir = ".\\format_data\\"
# full_path = base_dir+data_name
# model_path = base_dir+"example"
# if not os.path.exists(full_path):
#     copy_dirs(model_path, full_path)
# os.chdir(full_path)   # 改变当前路径到设置的data_name文件夹下
