import os
import shutil
import numpy as np
import random
from sys import argv

def makeQueryTrainData(queryClass):
    print('making the train data...')
    queryTrainDataPath1 = '/home/cloud/queryTrainData1_' + queryClass
    queryTrainDataPath2 = '/home/cloud/queryTrainData2_' + queryClass
    os.makedirs(queryTrainDataPath1 + '/' + queryClass)
    os.makedirs(queryTrainDataPath1 + '/' + 'other')
    os.makedirs(queryTrainDataPath2 + '/' + queryClass)
    os.makedirs(queryTrainDataPath2 + '/' + 'other')

    filePath1 = '/home/cloud/Dataset1'
    filePath2 = '/home/cloud/Dataset2'
    queryTrainDataNum1 = 0
    queryTrainDataNum2 = 0

    rate1 = '0.225 0.004 0.031 0.0 0.021 0.012 0.001 0.002 0.003 0.051 0.006 0.032 0.024 0.0 0.007 0.022 0.0 0.055 0.042 0.004 0.021 0.002 0.005 0.035 0.125 0.013 0.08 0.0 0.01 0.001 0.002 0.162'
    rate1 = rate1.split(' ')
    rate2 = '0.011 0.004 0.001 0.0 0.005 0.001 0.0 0.003 0.001 0.001 0.042 0.001 0.007 0 0.148 0.313 0.001 0.005 0.001 0.0 0.003 0.001 0.007 0.001 0.002 0.001 0.004 0.0 0.0 0.01 0.007 0.42'
    rate2 = rate2.split(' ')
    List = ['minivan','ambulance','beach_wagon','bicycle_build_for_two','cab','convertible','fire_engine','forklift','garbage_truck','jeep','jinrikisha','limousine','minibus','Model_T','moped','motor_scooter','mountain_bike','moving_van','pickup','police_van','racer','recreational_vehicle','snowmobile','snowplow','sports_car','streetcar','tow_truck','tractor','trailer_truck','tricycle','unicycle','other']

    rate = 0.3
    Total_num = 2000

    for Object in List:
        if Object != queryClass:
            file1 = os.listdir(filePath1 + '/' + Object)
            file2 = os.listdir(filePath2 + '/' + Object)
            sample1 = random.sample(file1, int(Total_num * (1-rate) *float(rate1[List.index(Object)])))
            sample2 = random.sample(file2, int(Total_num * (1-rate) *float(rate2[List.index(Object)])))
            for img1 in sample1:
                queryTrainDataNum1 += 1
                shutil.copy(filePath1 + '/' + Object + '/' + img1, \
                    queryTrainDataPath1 + '/' + 'other')
            for img2 in sample2:
                queryTrainDataNum2 += 1
                shutil.copy(filePath2 + '/' + Object + '/' + img2, \
                    queryTrainDataPath2 + '/' + 'other')

        else:
            file1 = os.listdir(filePath1 + '/' + Object)
            file2 = os.listdir(filePath2 + '/' + Object)
            sample1 = random.sample(file1, int(Total_num * rate))
            sample2 = random.sample(file2, int(Total_num * rate))
            for img1 in sample1:
                queryTrainDataNum1 += 1
                shutil.copy(filePath1 + '/' + Object + '/' + img1, \
                    queryTrainDataPath1 + '/' + Object)
            for img2 in sample2:
                queryTrainDataNum2 += 1
                shutil.copy(filePath2 + '/' + Object + '/' + img2, \
                    queryTrainDataPath2 + '/' + Object)

    print('done')
    print('Number of training data1: ',queryTrainDataNum1)
        