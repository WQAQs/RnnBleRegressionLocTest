import os
import pandas as pd
import globalConfig

def loadAllTxt2Csv(txt_rootdir, csv_rootdir):

    """
    :param txt_rootdir: string
        txt文件目录
    :param csv_rootdir: string
        生成的csv保存目录
    :return: nothing
    txt需用类别目录区分数据，生成的csv路径若目录不存在将创建对应目录
    转换的目标文件名会加上采集的时间戳防止重名
    """

    timed_dirs = os.listdir(txt_rootdir)  # 列出文件夹下所有的目录
    # for classDir in timed_dirs:
    #     classPath = os.path.join(txt_rootdir, classDir)  #每一个类别的基础路径
    #     dist_timed_path = os.path.join(csv_rootdir, classDir)  # 每一个类别生成的csv的基础路径
    #     if os.path.isdir(classPath):  # 对每个文件夹作为一个类别生成对应的csv
    #         files = os.listdir(classPath)
    #         for txt in files:
    #             # TODO 检查
    #             rawPoints_file_path = os.path.join(classPath, txt)
    #             txt2csv(rawPoints_file_path, dist_timed_path)

    for timed_dir in timed_dirs:
        timed_path = os.path.join(txt_rootdir, timed_dir)
        txt_files = os.listdir(timed_path)
        for txt_file in txt_files:
            dist_timed_path = os.path.join(csv_rootdir, timed_dir)  # 每一个类别生成的csv的基础路径
            # TODO 检查
            rawPoints_file_path = os.path.join(timed_path, txt_file)
            txt2csv(rawPoints_file_path, dist_timed_path)

def txt2csv(referenceRawPointFile, dist_dir):
    """
    单个txt文件为csv文件
    :param referenceRawPointFile: string
        txt文件路径
    :param dist_dir: string
        目标文件目录
    :return: nothing
    生成的文件名为txt文件名加生成的时间戳
    """
    txt_name = referenceRawPointFile.split("\\")[-1]  # 获取不带路径的txt文件名
    time = referenceRawPointFile.split("\\")[-2]  # 获取数据的采集日期
    rawPoints_file_name = txt_name.split('.')[0]  # 不带格式后缀的文件名
    newPoints_file_name = rawPoints_file_name + '_' + time + ".csv"  # 加上转换的时间戳
    dist_file_path = '\\'.join([dist_dir, newPoints_file_name])
    txt = pd.read_table(referenceRawPointFile, header=None, sep=',', names=['time', 'uuid', 'mac', 'rssi'])
    # txtDF = pd.DataFrame(data=txtDF, columns=['time', 'uuid', 'mac', 'rssi'])
    if not os.path.exists(dist_dir):  # 目标目录不存在时创建目录
        os.makedirs(dist_dir)
    txt.to_csv(dist_file_path, index=False, encoding='utf-8')

txt_rootdir = globalConfig.all_raw_txt_data_dir
csv_rootdir = globalConfig.all_unlabeled_csv_dir
loadAllTxt2Csv(txt_rootdir, csv_rootdir)