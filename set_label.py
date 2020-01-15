import os
import pandas as pd
import shutil
import globalConfig

point_label_map = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5,
                   '7':6, '8':7, '9':8, '10':9, '11':10, '12':11,
                   '13':12, '14':13, '15':14, '16':15, "17":16, '18':17,
                   '19':18, '20':19, '21':20}
labeld_csv_root_dir = globalConfig.all_labeled_csv_dir
unlabeled_csv_root_dir = globalConfig.all_unlabeled_csv_dir

def lable_data(source_root_dir, labeled_root_dir):
    timed_dirs = os.listdir(source_root_dir)
    for timed_dir in timed_dirs:
        timed_path = os.path.join(source_root_dir, timed_dir)
        csv_files = os.listdir(timed_path)
        for csv_file in csv_files:
            csv_file_path = os.path.join(timed_path, csv_file)
            lable_a_csv(csv_file_path, labeld_csv_root_dir)

def lable_a_csv(csv_file_path, labeled_root_dir):
    csv_name = csv_file_path.split("\\")[-1]  # 获取不带路径的txt文件名
    point_tag = csv_name.split('_')[1]
    class_tag = point_label_map[point_tag]
    new_dir = labeled_root_dir + "\\" + str(class_tag)
    new_file_path = new_dir + "\\" + csv_name
    if not os.path.exists(new_dir):  # 目标目录不存在时创建目录
        os.makedirs(new_dir)
    shutil.copyfile(csv_file_path, new_file_path)

lable_data(unlabeled_csv_root_dir, labeld_csv_root_dir)    # csv文件名字对应参考点的tag，csv文件的父文件夹名对应参考点的类tag
