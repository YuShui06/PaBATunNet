import matplotlib.pyplot as plt
import math

def drawAcc(history):
# 绘制训练精度以及验证精度
    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs,acc,'bo',label='Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
########################################################################################################

def drawLoss(history):
    ## 绘制训练损失和验证损失
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_valuse = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Train loss')
    plt.plot(epochs, val_loss_valuse, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
#########################################################################################################

## 计算RMSEC、RMSEP
def calculate_RMSE(p_value,r_value):
    cal_tem = 0
    cc_len = len(p_value)
    for i in range(cc_len):
        tem = math.pow((p_value[i] - r_value[i]),2)
        cal_tem = cal_tem + tem
    cal_cc = cal_tem/cc_len
    cal_fin = math.sqrt(cal_cc)
    return cal_fin
##########################################################################################################

## 计算决定系数R^2
def calculate_R2(p_value,r_value):
    average = r_value.mean(axis=0)
#     print("平均值：",average)
    cr_len = len(r_value)
    car_tem = 0
    cars_tem = 0
    for i in range(cr_len):
        temp_r = math.pow(r_value[i]-average,2)
        car_tem = car_tem + temp_r
    for i2 in range(cr_len):
        temp_s = math.pow(p_value[i2]-average,2)
        cars_tem = cars_tem + temp_s
    r_2 = (cars_tem/car_tem)
    # print("cr_len长度：",cr_len)
    # print("真值减平均：",car_tem)
    # print("预测值减平均：",cars_tem)
    # print("################")
    return r_2
#########################################################################################################

## 平滑曲线
def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
###############################################################################################################

def calculate_R21(p_value,r_value):
    average = r_value.mean(axis=0)
#     print("平均值：",average)
    cr_len = len(r_value)
    car_tem = 0
    cars_tem = 0
    for i in range(cr_len):
        temp_r = math.pow(r_value[i]-average,2)
        car_tem = car_tem + temp_r
    for i2 in range(cr_len):
        temp_s = math.pow(p_value[i2]-r_value[i2],2)
        cars_tem = cars_tem + temp_s
    r_2 = 1-(cars_tem/car_tem)
    # print("cr_len长度：",cr_len)
    # print("真值减平均：",car_tem)
    # print("预测值减平均：",cars_tem)
    # print("################")
    return r_2
###################################################################################################################
