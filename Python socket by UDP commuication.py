# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:54:47 2020

@author: ekdlw
"""
import socket

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

sock.close()