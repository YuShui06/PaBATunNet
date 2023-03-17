import ProgramGrainMoisture as PGM
import numpy as np
import pandas as pd
import datetime
import FunctionGrain as FC

# grain moisture
num = 1
epoch = 51
epochs_au = 100
batch_size_au = 1
max_r_2_p = 0.0
rmsec = 0.0
r_2_t = 0.0
rmsep = 0.0
r_2_p = 0.0
rec_rmsep = np.empty(num,dtype=float)
rec_r_2_p = np.empty(num,dtype=float)
rec_rmsec = np.empty(num,dtype=float)
rec_r_2_t = np.empty(num,dtype=float)

for i in range(1, epoch):
   now = datetime.datetime.now()
   now_s = now.strftime("%Y-%m-%d-%H-%M-%S")
   rec_rmsec, rec_r_2_t, rec_rmsep, rec_r_2_p = PGM.ModelGrainMoisture()
   write_data = [(i, now_s, epochs_au, batch_size_au, rec_rmsec, rec_r_2_t, rec_rmsep, rec_r_2_p)]  # 需要新写入的数据
   FC.write_To_Csv_Moisture(write_data)
   rmsec += rec_rmsec
   r_2_t += rec_r_2_t
   rmsep += rec_rmsep
   r_2_p += rec_r_2_p

avg_rmsec = rmsec / (epoch - 1)
avg_r_2_t = r_2_t / (epoch - 1)
avg_rmsep = rmsep / (epoch - 1)
avg_r_2_p = r_2_p / (epoch - 1)
df = pd.DataFrame(
   [('grain_moisture', now_s, epochs_au, epoch - 1, avg_rmsec, avg_r_2_t, avg_rmsep, avg_r_2_p)])  # 列表数据转为数据框
df.to_csv('../resultsdemo/allResults.csv', mode='a', index=False, header=False)
