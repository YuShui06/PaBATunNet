from keras import layers, Model
from keras.optimizers import RMSprop
import keras
import FunctionMilk as FC
import CommonFunction as CF
import tensorflow as tf
from keras.models import load_model
import datetime


def ModelMilkProtein():
    now = datetime.datetime.now()
    now_s = now.strftime("%Y-%m-%d-%H-%M-%S")
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    ## 准备参数
    epochs_au = 100
    batch_size_au = 1
    jihuo = 'tanh'


    callback_list_test =[
        keras.callbacks.ModelCheckpoint(
            filepath="./dataTimeMilkProtein/"+now_s+'.h5',      ##文件路径 存在当前路径下吧 还好找
            monitor='val_loss',         ## 监控指标
            save_best_only=True        ## 保持最佳模型
        )
    ]
    test_data, test_lable, train_data, train_lable = FC.getOdateMilkProtein()

    input = layers.Input(shape=(test_data.shape[1], 1), dtype='float32')
    net = layers.Conv1D(32, 3, activation=jihuo)(input)

    b_1 = layers.Conv1D(16, 7, activation=jihuo)(net)
    b_1 = layers.Conv1D(16, 5, activation=jihuo, padding='valid')(b_1)
    b_1 = layers.Conv1D(8, 3, activation=jihuo)(b_1)

    b_2 = layers.Conv1D(16, 5, activation=jihuo)(net)
    b_2 = layers.Conv1D(16, 3, activation=jihuo, padding='valid')(b_2)
    b_2 = layers.Conv1D(8, 3, activation=jihuo)(b_2)

    b_3 = layers.Conv1D(16, 7, activation=jihuo)(net)
    b_3 = layers.Conv1D(16, 5, activation=jihuo, padding='valid')(b_3)
    b_3 = layers.Conv1D(8, 3, activation=jihuo)(b_3)

    b_4 = layers.AvgPool1D(3)(net)
    b_4 = layers.Conv1D(16, 7, activation=jihuo)(b_4)
    b_4 = layers.Conv1D(16, 5, activation=jihuo, padding='valid')(b_4)
    b_4 = layers.Conv1D(8, 3, activation=jihuo)(b_4)

    b_5 = layers.AvgPool1D(3)(net)
    b_5 = layers.Conv1D(16, 5, activation=jihuo)(b_5)
    b_5 = layers.Conv1D(16, 1, activation=jihuo, padding='valid')(b_5)
    b_5 = layers.Conv1D(8, 3, activation=jihuo)(b_5)

    net = layers.concatenate([b_1, b_2, b_3, b_4, b_5], axis=1)
    output = layers.Flatten()(net)
    output = layers.Dense(4)(output)
    output = layers.Dense(4)(output)
    output = layers.Dense(2)(output)
    output = layers.Dense(1)(output)
    model = Model(input, output)

    model.summary()
    model.compile(optimizer=RMSprop(),loss='mse')
    history = model.fit(train_data,train_lable,
                            epochs=epochs_au,
                            batch_size=batch_size_au,
                            validation_data=(test_data,test_lable),
                            callbacks= callback_list_test
                            )

    CF.drawLoss(history)  ## 绘制当前的验证曲线

    model = load_model("./dataTimeMilkProtein/"+now_s+'.h5')
    result_trian = model.predict(train_data)
    result_predict = model.predict(test_data)
    rmsec = CF.calculate_RMSE(result_trian,train_lable) ## 训练集上的RMSE
    rmsep = CF.calculate_RMSE(result_predict,test_lable)  ## 测试集上的RMSE
    r_2_t = CF.calculate_R21(result_trian,train_lable)## 训练集上的R_2
    r_2_p = CF.calculate_R21(result_predict,test_lable)## 测试集上得R_2

    return rmsec, r_2_t, rmsep, r_2_p
