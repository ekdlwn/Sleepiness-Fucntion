"""Example program to show how to read a multi-channel time series from LSL."""
import pandas as pd
import socket
from pylsl import StreamInlet, resolve_stream

"""
# 시뮬레이션 UDP 신호 입력 받아서 데이터 저장받기 위한 신호 코스
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

port = 4000
sock.bind(("127.0.0.1", port))
print('The server is ready to receive')

while True:
    data, addr = sock.recvfrom(2048) # 버퍼데이터 크기 설정 및 데이터 입력부
    modifiedData = data.decode().upper() # 통신 받은 데이터 해독
    
    modifiedData=modifiedData.lstrip('"",') # string 데이터 중 불필요한 부분 제거
    modifiedData=modifiedData.rstrip(',""')
    modifiedData=float(modifiedData) # string to float
    
    print(modifiedData)
    
    if modifiedData == 'START':
        EEGDATA()
    
sock.close()
"""



def EEGDATA():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    
    # 빈 Dataframe 만들기
    Data_temp = pd.DataFrame()
    Data_all = pd.DataFrame()
    
    # 데이터 저장 주요 변수들
    Frame_rate = 500
    Data_Save_time = 5
    Data_index = ["Fp1", "Fp2", "F3", "F4", "P4", "P3", "O1", "O2", "Exg1", "Exg2", "Exg3", "Exg4", "ECG", "Resp", "PPG", "SpO2", "HR", "GSR", "Tep"]
    
    # 실험자 변수
    Tester_sex = 0
    Tester_age = 20
    Tester_driverLicense = 1
    Tester_sleepTime = 2407
    Tester_testtime = 13
    
    # 폴더 번호 설정
    i = 0
    
    # 데이터 저장 위치
    Folder = 'C:\\Users\\ekdlw\\Desktop\\TEST\\Bio'
    
    
    # 데이터 저장 시간
    Cutting_time = Frame_rate * Data_Save_time
    
    
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        # 데이터를 입력한 후, 쭉 나열하기
        sample, timestamp = inlet.pull_sample()
        
        # Timestamp 넣기
        sample.insert(0,timestamp)
        
        Data_temp = pd.DataFrame(sample)
        Data_temp = Data_temp.T
        
        Data_all = pd.concat([Data_all, Data_temp])
        len_Data_all = len(Data_all)
        if len_Data_all % Cutting_time == 0:
            
            # File name 정의
            File_name = f'\\{Tester_sex}{Tester_age}{Tester_driverLicense}{Tester_sleepTime}{Tester_testtime}_{i}.csv'
            
            Data_all.to_csv(Folder+File_name)
            
            # 파일 숫자 늘리기
            i = i+1
            
            # 데이터 비우기
            Data_all = pd.DataFrame()
        
        print(timestamp, sample)
        
        
EEGDATA()