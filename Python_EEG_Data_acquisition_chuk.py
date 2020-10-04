"""Example program to show how to read a multi-channel time series from LSL."""
import pandas as pd
from pylsl import StreamInlet, resolve_stream

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

Data_temp = pd.DataFrame()
Data_all = pd.DataFrame()

while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    # 데이터를 입력한 후, 쭉 나열하기
    sample, timestamp = inlet.pull_chunk(max_samples=500)
    Data_temp = pd.DataFrame(sample)
    
    Data_all = pd.concat([Data_all, Data_temp])
    
    #if Cutting_time ==
    
    print(timestamp, sample)
    
    
    
