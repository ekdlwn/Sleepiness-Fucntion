import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 빈 Dataframe 만들기
Data_temp = pd.DataFrame()
Data_all = pd.DataFrame()

# 데이터 저장 주요 변수들
Frame_rate = 500
Data_Save_time = 5
Data_index = ["Fp1", "Fp2", "F3", "F4", "P4", "P3", "O1", "O2", "Exg1", "Exg2", "Exg3", "Exg4", "ECG", "Resp", "PPG", "SpO2", "HR", "GSR", "Tep"]

# 실험자 변수
Tester_sex = 1
Tester_age = 23
Tester_driverLicense = 0
Tester_sleepTime = 2407
Tester_testtime = 19

# 폴더 번호 설정
i = 0

# 데이터 저장 위치
Folder = r'C:\Users\ekdlw\Desktop\Bio Data Analysis\CSV_file'
File_name = f'\\{Tester_sex}{Tester_age}{Tester_driverLicense}{Tester_sleepTime}{Tester_testtime}_{i}.csv'


# 데이터 저장 시간
Cutting_time = Frame_rate * Data_Save_time

data, header = pyxdf.load_xdf(r'C:\Users\ekdlw\Desktop\Bio Data Analysis\XDF_file\20101001.xdf')

for stream in data:
    i = i+1
    y = stream['time_series']
    x = stream['time_stamps']
    
    df1 = pd.DataFrame(y)
    df2 = pd.DataFrame(x)
    
    df1 = pd.concat([df1,df2],axis=1)
    
    File_name = f'\\{Tester_sex}{Tester_age}{Tester_driverLicense}{Tester_sleepTime}{Tester_testtime}_{i}.csv'
    df1.to_csv(Folder+File_name, index = False, columns = False)
    
    if isinstance(y, list):
        # list of strings, draw one vertical line for each marker
        
        for timestamp, marker in zip(stream['time_stamps'],y):
            plt.axvline(x=timestamp)
            print(f'Marker "{marker[0]}",@ {timestamp:.2f}s')
    elif isinstance(y, np.ndarray):
        # numeric data, draw as lines
        plt.plot(stream['time_stamps'], y)
        
    else:
        raise RuntimeError('Unknown stream format')
            
