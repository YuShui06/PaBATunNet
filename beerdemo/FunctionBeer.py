import pandas as pd
import scipy.io as scio
import PretreatmentFunction as PF

def getOdateBeerYeast():
    dataFile = '../data/'
    dataName = 'beer_yeast.mat'
    data = scio.loadmat(dataFile + dataName)
    datax_train = data['Xcal']
    datax_test = data['Xtest']
    datay_train = data['YCal']
    datay_test = data['Ytest']
    datax_train = PF.D1(datax_train)
    datax_test = PF.D1(datax_test)
    a = len(datax_train)
    b = len(datax_test)
    datax_train -= datax_train.mean(axis=0)
    datax_test -= datax_test.mean(axis=0)
    datax_train /= datax_train.std(axis=0)
    datax_test /= datax_test.std(axis=0)
    datax_train = datax_train.astype('float32')
    datax_test = datax_test.astype('float32')
    datay_train = datay_train.astype('float32')
    datay_test = datay_test.astype('float32')
    datax_train = datax_train.reshape(a, 576)
    datax_test = datax_test.reshape(b, 576)

    return datax_test, datay_test, datax_train, datay_train

## 向CSV中写入数据
def write_To_Csv_Yeast(write_data):
    df = pd.DataFrame(write_data,
                      columns=['num', 'model_name', 'epochs', 'batch_size', 'RMSEC', 'R_C', 'RMSEP', 'R_P'])  # 列表数据转为数据框
    df.to_csv('dataResultsBeerYeast.csv', mode='a', index=False, header=False)
    return


















