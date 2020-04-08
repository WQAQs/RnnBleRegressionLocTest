import dataProcess as dp
import globalConfig  #不能删除,因为在 globalConfig.py 中更改了当前路径


pointTxtRootDir = globalConfig.root_txt_dir  # 原始数据存在的文件夹
pointCsvRootDir = globalConfig.root_csv_dir  # 转换的csv文件目标文件夹，也作为合并csv的源路径
ibeaconFilePath = globalConfig.ibeaconFilePath  # ibeacon统计文件目标路径及文件名
allPointCsvRootDir = globalConfig.generate_sampleset_all_labeled_csv_dir  # 总数据数据文件夹

# dp.loadAllTxt2Csv(pointTxtRootDir, pointCsvRootDir)  # 将原始数据加载为Csv文件

# 修改后若无单独保存csv数据需求，直接将生成的csv文件保存到总数据目录中，无需额外合并
# dp.mergeAllCsv(allPointCsvRootDir, pointCsvRootDir)  # 将生成的Csv文件加入总数据

# dp.updateAllIbeaconDataSet(allPointCsvRootDir, ibeaconFilePath)  # 更新ibeaconDataSet

dp.createSampleDataSet(allPointCsvRootDir)  # 创建样本集
