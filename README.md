# RnnBleRegressionLocTest
每个timestamp输入采集到的一个ibeacon强度值，在 m 个timestamp后输出定位结果（x，y 坐标）
实验记录：
4days 数据训练模型   best to mseloss 0.28
10days 数据训练模型   best to mseloss 0.21
绘制了误差距离的CDF曲线
（但上面的模型都用了2019.12.11 和 2019.12.12 的错误数据参与训练，需要重新再训练试试）