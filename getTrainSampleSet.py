import dataProcess as dp
import globalConfig  #不能删除,因为在 globalConfig.py 中更改了当前路径

'''新代码中以配置好默认路径，只需修改globalConfig.py，若无特殊需求不要改此处变量'''
pointTxtRootDir = '.\\raw_data\\train'  # 原始数据存在的文件夹
pointCsvRootDir = '.\\points_csv\\train'  # 转换的csv文件目标文件夹，也作为合并csv的源路径
ibeaconFilePath = '.\\ibeacon_mac_count_day1217.csv'  # ibeacon统计文件目标路径及文件名
allPointCsvRootDir = '.\\new_labeled'  # 总数据数据文件夹
# 样品数据集目标路径，可以设置为训练集或测试集
dp.sampleDataSetFilePath = '.\\rnn_sample_set_onehot_test2.csv'


# dp.loadAllTxt2Csv(pointTxtRootDir, pointCsvRootDir)  # 将原始数据加载为Csv文件

# 修改后若无单独保存csv数据需求，直接将生成的csv文件保存到总数据目录中，无需额外合并
# dp.mergeAllCsv(allPointCsvRootDir, pointCsvRootDir)  # 将生成的Csv文件加入总数据

# dp.updateAllIbeaconDataSet(allPointCsvRootDir, ibeaconFilePath)  # 更新ibeaconDataSet

dp.createSampleDataSet(allPointCsvRootDir, ibeaconFilePath)  # 创建样本集
