import globalConfig
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

valid_ibeacon_file = globalConfig.valid_ibeacon_file
# wordvec_file = globalConfig.wordvec_file
# valid_ibeacon_file = ".\\valid_ibeacon_mac.csv"
n_valid_ibeacon = -1
word2id_map = {}
id2word_map = {}
id2wordvec_map = {}
onehot_mac_dic = {}
wordvec = []  # 没有归一化的处理的
wordvec_uniformed = []  # 把 rssi 除以128 作归一化的处理后的

def onehot(ibeacon_csv):
    global n_valid_ibeacon
    mac_df = ibeacon_csv.mac
    mac_array = np.asarray(mac_df)
    n_valid_ibeacon = mac_array.shape[0]
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(mac_array)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    for i in range(mac_array.shape[0]):
        onehot_mac_dic[mac_array[i]] = onehot_encoded[i]


def get_onehot(mac):
    global onehot_mac_dic
    if mac in onehot_mac_dic:
        onehot_mac = onehot_mac_dic[mac]
        # str_onehot = ""
        # for i in onehot_mac:
        #     str_onehot = str_onehot + str(i)+"|"
        # result =str_onehot.rstrip("|")
        onehot_mac = onehot_mac.tolist()
        return onehot_mac
    else:
        return None

def generate_word_id_wordvec_map():
    global word2id_map
    global id2word_map
    global id2wordvec_map
    global wordvec_uniformed
    global wordvec
    id = 0
    ibeacon_csv = pd.read_csv(valid_ibeacon_file)
    onehot(ibeacon_csv)
    # 作为padding的word
    padding = "padding"
    word2id_map[padding] = id
    id2word_map[str(id)] = padding
    vec = [0 for i in range(n_valid_ibeacon + 1)]
    id2wordvec_map[str(id)] = vec
    wordvec_uniformed.append(vec)
    wordvec.append(vec)
    id += 1
    for row in ibeacon_csv.itertuples():
        mac = row.mac
        for value in range(-40, -128, -1):
            mac_value = mac + '_' + str(value)  # mac_value是mac和该mac对应的信号强度value的组合
            word2id_map[mac_value] = id # mac_value表示一个word
            id2word_map[str(id)] = mac_value
            mac_rssi = get_onehot(mac)
            mac_uniform_rssi = get_onehot(mac)
            mac_rssi.append(value)
            mac_uniform_rssi.append(value/128.0)
            mac_rssi = [np.int64(i) for i in mac_rssi]
            id2wordvec_map[str(id)] = mac_rssi
            wordvec_uniformed.append(mac_uniform_rssi)
            wordvec.append(mac_rssi)
            id += 1


def generateWordVec():
    global wordvec_file


generate_word_id_wordvec_map()

len1 = len(wordvec)
len2 = len(id2wordvec_map)
len3 = len(word2id_map)
res = id2wordvec_map