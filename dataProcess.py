import numpy as np
import pandas as pd
import os
import time
import shutil
import globalConfig
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import getWordVector
from sklearn.utils import shuffle

# 样品数据集目标路径，可以设置为训练集或测试集
# sampleDataSetFilePath = '.\\sample_train.csv'
sampleDataSetFilePath = globalConfig.sample_dataset_file
sample_count = 0  # 数据量统计
initVal = -128  # 维度数据初始化
timeInterval = globalConfig.timeInterval # 取样时间间隔
valid_ibeacon_end_index = 0  # 作为可用维度的mac地址在ibeacon统计中的下标
WholeTimeInterval = 1000 * 360  # 暂时未用到
# offset = 0  # TODO 含义？
# stableCount = globalConfig.stableCount  # 维度设置，暂时改为直接确定选取的mac数目
ibeacon_column_tags = ["scan_index", "count", "mac", "uuid", ]  # ibeacon列标签
ibeacon_dataset = pd.DataFrame(columns=ibeacon_column_tags)
out_file_created = os.path.exists(sampleDataSetFilePath)  # 标记目标文件是否已创建

valid_ibeacon_file = globalConfig.valid_ibeacon_file
train_dataset_file = globalConfig.train_dataset_file
valid_dataset_file = globalConfig.valid_dataset_file

def loadAllTxt2Csv(txt_rootdir, csv_rootdir):

    """
    :param txt_rootdir: string
        txt文件目录
    :param csv_rootdir: string
        生成的csv保存目录
    :return: nothing
    txt需用类别目录区分数据，生成的csv路径若目录不存在将创建对应目录
    转换的目标文件名会加上时间戳（转换日期）防止重名
    """

    paths = os.listdir(txt_rootdir)  # 列出文件夹下所有的目录
    for classDir in paths:
        classPath = os.path.join(txt_rootdir, classDir)  #每一个类别的基础路径
        dist_class_path = os.path.join(csv_rootdir, classDir)  # 每一个类别生成的csv的基础路径
        if os.path.isdir(classPath):  # 对每个文件夹作为一个类别生成对应的csv
            files = os.listdir(classPath)
            for txt in files:
                # TODO 检查
                rawPoints_file_path = os.path.join(classPath, txt)
                txt2csv(rawPoints_file_path, dist_class_path)


def mergeAllCsv(original_csv_rootdir, plus_csv_rootdir):
    """
    将新表格加入总数据的方法
    :param original_csv_rootdir: string
        合并的目标目录
    :param plus_csv_rootdir: string
        合并入总数据的新数据源目录（两个目录均不包含类别目录）
    :return: noting
    注意目标路径需包含所有可能的类目录，若不存在将报错
    """
    dirs = os.listdir(plus_csv_rootdir)  # 列出文件夹下所有的目录与文件  reference_point_count
    for class_dir in dirs:
        class_base_dir = os.path.join(plus_csv_rootdir, class_dir)
        dist_path = os.path.join(original_csv_rootdir, class_dir)
        if os.path.isdir(class_base_dir):
            csvs = os.listdir(class_base_dir)  # 对所有类别目录下的文件进行拷贝
            for csv in csvs:
                csv_path = os.path.join(class_base_dir, csv)
                dist_path = os.path.join(dist_path, csv)  # 目标文件路径
                shutil.copy(csv_path, dist_path)  # 此处若目标路径不存在将报错


def updateAllIbeaconDataSet(csv_rootdir, ibeacon_file_path):
    """
    更新ibeacon统计表的方法
    :param csv_rootdir: string
        csv源目录，使用路径下所有类别数据统计mac
    :param ibeacon_file_path:
        生成的ibeacon统计表，包含文件名，若原文件存在将被替换
    :return: noting

    """
    dirs = os.listdir(csv_rootdir)  # 列出文件夹下所有的目录与文件
    for csv_dir in dirs:
        class_dir = os.path.join(csv_rootdir, csv_dir)
        files = os.listdir(class_dir)
        for file in files:
            file_path = os.path.join(class_dir, file)
            data = pd.read_csv(file_path, names=['time', 'uuid', 'mac', 'rssi'])
            updateIbeaconDataSet(data)
    ibeacon_dataset.sort_values("count", inplace=True, ascending=False)
    ibeacon_dataset.to_csv(ibeacon_file_path, index=0)  # 保存ibeacon_dataset到csv文件
    ibeacon_csv = pd.read_csv(ibeacon_file_path)
    print('ibeacon_csv:')
    print(ibeacon_csv)
    print(ibeacon_csv.dtypes)

def divide_sample_dataset(sample_dataset):
    train_dataset = sample_dataset.sample(frac=0.8, random_state=0)
    valid_dataset = sample_dataset.drop(train_dataset.index)
    train_dataset.to_csv(train_dataset_file, index=False, encoding='utf-8')
    valid_dataset.to_csv(valid_dataset_file, index=False, encoding='utf-8')

def load_dataset(dataset):

    reference_tag = dataset.values[:, 0]
    data_input = dataset.values[:, 5] #包括index=5
    # data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    coordinates = dataset.values[:, 1:3] #包括index=1，不包括index=3
    return data_input, coordinates, reference_tag

def load_data(data_file, divide_dataset_flag):
    dataset = pd.read_csv(data_file)
    shuffled_dataset = shuffle(dataset)
    if divide_dataset_flag:
        divide_sample_dataset(shuffled_dataset)
    train_dataset = pd.read_csv(train_dataset_file)
    valid_dataset = pd.read_csv(valid_dataset_file)
    train_input, train_coordinates, train_reference_tag = load_dataset(train_dataset)
    valid_input, valid_coordinates, valid_reference_tag = load_dataset(valid_dataset)
    train_coordinates = train_coordinates.astype(float)
    valid_coordinates = valid_coordinates.astype(float)
    return train_input, train_coordinates, train_reference_tag, valid_input, valid_coordinates, valid_reference_tag

# generate sample dataset
def createSampleDataSet(pointCsvRootDir, shuffle_flag=True):
    """
    创建数据样本集的方法，通过ibeacon统计表筛选维度并
    :param pointCsvRootDir: string
        csv数据目录
    :param ibeaconFilePath:
        ibeacon统计表文件路径
    :return:
    """
    # ibeacon_csv = pd.read_csv(ibeaconFilePath)
    column_tags = configColumnTags()
    loadAllCsvFile2SampleSet(pointCsvRootDir, column_tags)
    dataset = pd.read_csv(globalConfig.sample_dataset_file)
    shuffled_dataset = shuffle(dataset)
    if shuffle_flag:
        divide_sample_dataset(shuffled_dataset)


def updateIbeaconDataSet(point_data):
    """
    根据数据更新ibeacon表的方法
    :param point_data: DataFrame or TextParser
    :return: nothing

    """
    global ibeacon_dataset
    groupby_data = point_data.groupby(['mac'])  # 按照id进行分类
    for mac_value, group_data in groupby_data:
        print("group_data: \n{}".format(group_data))
        print("group_data.count(): \n{}".format(group_data.count()))
        print("group_data['mac'].count()\n{}".format(group_data['mac'].count()))
        macDF = ibeacon_dataset[ibeacon_dataset['mac'].isin([mac_value])]
        mac_count = group_data['mac'].count()
        if macDF.empty == True:
            data = ({'count': mac_count, 'mac': mac_value, 'uuid': -1, },)
            mac_dataDF = pd.DataFrame(data)
            ibeacon_dataset = ibeacon_dataset.append(mac_dataDF, ignore_index=True)
        else:
            index = macDF.index.values[0]
            ibeacon_dataset.loc[index:index, 'count'] = macDF['count'] + mac_count


def configColumnTags():
    """
    确定样本集维度标签的方法
    :param ibeacon_csv: DataFrame or TextParser
    ibeacon
    :return: list<string>
    """
    column_tags = ["reference_tag", "coordinate_x", "coordinate_y", "cluster_tag", "direction_tag", "onehotmac_rssi_sentence"]
    # for i in range(globalConfig.n_timestamps):
    #     column_tags.append("rssi")
    # column_tags = ["reference_tag", "coordinate_x", "coordinate_y", "cluster_tag", "direction_tag"]
    # for index, row in ibeacon_csv.iterrows():
    #     if index >= stableCount:  # 若该行的mac出现次数小于阈值则不计入维度
    #         break
    #     #tag = "mac%drssi" % index
    #     tag = row["mac"]  # 这里改为直接使用mac地址作为维度标签以减少使用ibeacon统计表的次数
    #     column_tags.append(tag)
    # columns_count = len(column_tags)  # 加上 参考的标签 和 采样方向 两列
    # print("colums_count:{}".format(columns_count))
    return column_tags


def txt2csv(referenceRawPointFile, dist_dir):
    """
    转换txt格式到csv格式文件
    :param referenceRawPointFile: string
        txt文件路径
    :param dist_dir: string
        目标文件目录
    :return: nothing
    生成的文件名为txt文件名加生成的时间戳
    """
    txt_name = referenceRawPointFile.split("\\")[-1]  # 获取不带路径的txt文件名
    rawPoints_file_name = txt_name.split('.')[0]  # 不带格式后缀的文件名
    newPoints_file_name = rawPoints_file_name + time.strftime("_%Y_%m_%d", time.localtime()) + ".csv"  # 加上转换的时间戳
    dist_file_path = '\\'.join([dist_dir, newPoints_file_name])
    txt = pd.read_table(referenceRawPointFile, header=None, sep=',', names=['time', 'uuid', 'mac', 'rssi'])
    # txtDF = pd.DataFrame(data=txtDF, columns=['time', 'uuid', 'mac', 'rssi'])
    if not os.path.exists(dist_dir):  # 目标目录不存在时创建目录
        os.makedirs(dist_dir)
    txt.to_csv(dist_file_path, index=False, encoding='utf-8')


def deprecated_sliceAndAverage(referencePoint_csv, reference_tag, cluster_tag, direction_tag, coordinate_x,coordinate_y,column_tags, samples_dataset):
    """
    将数据按设定的时间片分割并求平均的方法
    :param referencePoint_csv: DataFrame
        读取的待处理数据
    :param ibeacon_csv: DataFrame
        ibeacon统计表数据
    :param reference_tag: string
        类型标签
    :param direction_tag: string
        方向标签
    :param column_tags: list<string>
        维度标签
    :return: number
        返回生成的数据条数
    """
    i = 0
    j = 0
    # 访问csv文件的每一行的数据，每一个循环里处理 1s 的数据，
    # 每轮循环结束更新 i 的值时，注意要让它指向下一秒的数据起点
    rownum = referencePoint_csv.shape[0]  # 取数据行数
    tag_map = {}
    for idx in range(len(column_tags)):  # 使用dict提高查找下标的速度
        tag_map[column_tags[idx]] = idx
    while i < rownum:
        # if referencePoint_csv['time'][i] > referencePoint_csv['time'][0] + WholeTimeInterval / 2:
        #     break
        macs_rssi = [initVal for i in range(len(column_tags))]  # 用固定值（-128）初始化列表
        time_col = referencePoint_csv['time']
        while j < rownum and time_col[j] < time_col[i] + timeInterval:
            j += 1
        if j >= rownum:  # 移除文件末尾不足一秒的数据
            break
        # 同一 mac 地址出现多次，则对多次的 rssi 求平均
        groupby_data = referencePoint_csv.iloc[i: j].groupby(['mac'])  # 按照mac进行分类
        for mac_value, group_data in groupby_data:
            tag_idx = tag_map.get(mac_value, -1)
            if tag_idx > -1:
                rrsi_mean = group_data['rssi'].mean()
                macs_rssi[tag_idx] = rrsi_mean
        ####  参考点标签 和  采样方向标签  赋值给 macs_rssi[index] #####
        # column_tags 前5列分别是："reference_tag", "coordinate_x", "coordinate_y", "cluster_tag", "direction_tag"
        macs_rssi[0] = reference_tag
        macs_rssi[1] , macs_rssi[2] = coordinate_x,coordinate_y
        macs_rssi[3] = cluster_tag
        macs_rssi[4] = direction_tag
        macs_rssi = np.array(macs_rssi).reshape(1, len(macs_rssi))

        macs_rssiDF = pd.DataFrame(data=macs_rssi, columns=column_tags)
        #### 将这 1s 的样本加入到样本集中  ####
        samples_dataset = samples_dataset.append(macs_rssiDF, ignore_index=True)
        i = j  # 更新 i 的值，让它指向下一秒的数据起点
    return samples_dataset

def config_coordinate(reference_tag):
    reference_point_csv = pd.read_csv(globalConfig.reference_points_coordinates_file)
    df = reference_point_csv[reference_point_csv['reference_tag'].isin([reference_tag, ])]
    x = df['coordinate_x'].values  # 返回的x 和 y 都分别是一个list
    y = df['coordinate_y'].values
    return x[0], y[0]  # 所以后面用到它们的值要返回 x[0] y[0]


def csv2sample_data(referencePoint_csv, reference_tag, coordinate_x, coordinate_y, cluster_tag,direction_tag,column_tags,timeInterval):
    all_samples = []
    one_sample = []  # 记录每一个样本包括一些标签数据和多个mac的信号强度
    one_sample_mac_rssi = []  # 记录每一个样本中的多个mac的信号强度
    # 访问csv文件的每一行的数据，每一个循环里处理 1s 的数据，
    # 每轮循环结束更新 i 的值时，注意要让它指向下一秒的数据起点
    begin_time = referencePoint_csv.iloc[0][0]  # 分段的开始时间
    for row in referencePoint_csv.itertuples():
        ######### 数据超出一个样本的时间，就保存一个样本到样本集 #########
        if row.time > begin_time + timeInterval:
            one_sample.append(reference_tag)
            one_sample.append(coordinate_x)
            one_sample.append(coordinate_y)
            one_sample.append(cluster_tag)
            one_sample.append(direction_tag)
            one_sample.append(one_sample_mac_rssi)
            all_samples.append(one_sample)  # 将一个样本加入样本集
            one_sample = []
            one_sample_mac_rssi = []
            begin_time = row.time  # 重置时间段的开始值
        ###### 在一个样本的时间内，更新记录 one_sample_mac_rssi的数据 #######
        onehot_mac = getWordVector.get_onehot(row.mac)
        if onehot_mac is None:
            continue
        else:
            word = row.mac + '_' + str(row.rssi)
            id = getWordVector.word2id_map[word]
            one_sample_mac_rssi.append(id)

    samples_dataset = pd.DataFrame(all_samples, columns=column_tags)
    return samples_dataset

def loadAllCsvFile2SampleSet(csv_rootdir, column_tags):
    """
    加载所有csv文件数据到样本集
    :param csv_rootdir: string
        csv文件根目录，对应子目录名为数据类别标签
    :param column_tags: list<string>
        样本维度列表
    :return: nothing
    """
    global out_file_created
    samples_dataset = pd.DataFrame(columns=column_tags)
    samples_count = 0
    # 初始化样本集 samples_dataset
    # 设置样本集的列数（包括 特征维度 和 参考点标签）和列名
    dirs = os.listdir(csv_rootdir)  # 列出文件夹下所有的目录与文件
    for cluster_dir in dirs:
        class_base_path = os.path.join(csv_rootdir, cluster_dir)
        files = os.listdir(class_base_path)
        cluster_tag = cluster_dir
        print("cluster_tag = {}".format(cluster_tag))
        for file in files:
            print("--加载csv文件{}到SampleSet:".format(file))
            file_path = os.path.join(class_base_path, file)  # file 是带文件格式后缀的文件名
            file_name = file.split('.')[-2]  # 不带文件格式后缀的文件名
            direction_tag = file_name.split('_')[0]
            reference_tag = file_name.split('_')[1]
            # 加载参考点的csv文件和ibeacon的csv文件
            csv = pd.read_csv(file_path)
            #ibeacon_csv = pd.read_csv(ibeacon_file_path)
            # 设置 参考点标签referencePoint_tag 和 TODO (采样方向标签referencePoint_sample_direction_tag暂未设置）
            coordinate_x, coordinate_y = config_coordinate(reference_tag)
            samples_dataset = csv2sample_data(csv, reference_tag, coordinate_x, coordinate_y, cluster_tag,
                                              direction_tag, column_tags, globalConfig.timeInterval)
            if out_file_created:
                samples_dataset.to_csv(sampleDataSetFilePath, index=False, encoding='utf-8', header=False, mode="a") #接着在文件末尾添加数据，即以append方式写数据

            else:
                samples_dataset.to_csv(sampleDataSetFilePath, index=False, encoding='utf-8')
                out_file_created = True

    print('sliceAndAverage 外1的 samples_dataset:')
    print(samples_dataset)
    print('samples_count:')
    print(samples_dataset.shape[0])






