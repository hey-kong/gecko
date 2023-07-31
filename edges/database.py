import paho.mqtt.client as mqtt
import os
import numpy as np
import pickle
import sqlite3
import logging 
logging.basicConfig(level=logging.WARNING) 
logger = logging.getLogger(__name__)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: "+str(rc))
    client.subscribe('data/+',2)

def on_message(client, userdata, message):
    print('got one!')
    writeData(message.topic,message.payload)

def on_subscribe(client, userdata, mid, granted_qos): 
    print('subscribe data/+ success')

def writeData(topic,payload):
    if topic[5:] == 'stop':
        parentDirectory = '/home/edge/npy/'
        tailDirectory = '_Eyedge.npy'
        np.save(parentDirectory + 'MNv2ReferTime1' + tailDirectory, np.array(MNv2ReferTime1))
        np.save(parentDirectory + 'MNv2ReferTime2' + tailDirectory,np.array(MNv2ReferTime2))
        np.save(parentDirectory + 'MNv2ReferTime3' + tailDirectory,np.array(MNv2ReferTime3))
        np.save(parentDirectory + 'RN152ReferTime' + tailDirectory,np.array(RN152ReferTime))
        np.save(parentDirectory + 'PublishTime_ContourPic' + tailDirectory,np.array(PublishTime_ContourPic))
        np.save(parentDirectory + 'MNv2_Queue_Length1' + tailDirectory,np.array(MNv2_Queue_Length1))
        np.save(parentDirectory + 'MNv2_Queue_Length2' + tailDirectory,np.array(MNv2_Queue_Length2))
        np.save(parentDirectory + 'MNv2_Queue_Length3' + tailDirectory,np.array(MNv2_Queue_Length3))
        np.save(parentDirectory + 'RN152_Queue_Length' + tailDirectory,np.array(RN152_Queue_Length))
        print('npy saved!')
    else:
        try:
            conn = sqlite3.connect('/home/edge/nodesdata.db')
            cursor = conn.cursor()
            #cursor.execute('select value from nodesdata where name=\'' + topic[5:] + '\'')
            cursor.execute('select * from nodesdata')
            values = cursor.fetchall()
            MNv2ReferTime1_value = float(values[0][1])
            MNv2ReferTime2_value = float(values[1][1])
            MNv2ReferTime3_value = float(values[2][1])
            RN152ReferTime_value = float(values[3][1])
            PublishTime_ContourPic_value = float(values[4][1])
            MNv2_Queue_Length1_value = float(values[5][1])
            MNv2_Queue_Length2_value = float(values[6][1])
            MNv2_Queue_Length3_value = float(values[7][1])
            RN152_Queue_Length_value = float(values[8][1])
            alpha_value = float(values[9][1])
            payload = float(payload)
            query_interval = 0.6
            gamma1 = 0.03
            rate = 0.2

            if topic[5:] == 'MNv2ReferTime1':
                MNv2ReferTime1.append(payload)
                MNv2ReferTime1_value = payload * rate + MNv2ReferTime1_value * (1 - rate)
                payload = MNv2ReferTime1_value

            if topic[5:] == 'MNv2ReferTime2':
                MNv2ReferTime2.append(payload)
                MNv2ReferTime2_value = payload * rate + MNv2ReferTime2_value * (1 - rate)
                payload = MNv2ReferTime2_value

            if topic[5:] == 'MNv2ReferTime3':
                MNv2ReferTime3.append(payload)
                MNv2ReferTime3_value = payload * rate + MNv2ReferTime3_value * (1 - rate)
                payload = MNv2ReferTime3_value

            if topic[5:] == 'RN152ReferTime':
                RN152ReferTime.append(payload)
                RN152ReferTime_value = payload * rate + RN152ReferTime_value * (1 - rate)
                payload = RN152ReferTime_value

            if topic[5:] == 'PublishTime_ContourPic':
                PublishTime_ContourPic.append(payload)
                PublishTime_ContourPic_value = payload * rate + PublishTime_ContourPic_value * (1 - rate)
                payload = PublishTime_ContourPic_value
                
            if topic[5:] == 'MNv2_Queue_Length1':
                MNv2_Queue_Length1.append(payload)

            if topic[5:] == 'MNv2_Queue_Length2':
                MNv2_Queue_Length2.append(payload)

            if topic[5:] == 'MNv2_Queue_Length3':
                MNv2_Queue_Length3.append(payload)

            if topic[5:] == 'RN152_Queue_Length':
                RN152_Queue_Length.append(payload)

            cursor.execute('update nodesdata set value=\'' + str(payload) + '\' where name=\'' + topic[5:] + '\'')
            cursor.execute('select value from nodesdata where name=\'' + topic[5:] + '\'')
            values = cursor.fetchall()
            print(topic[5:] + ': ' + values[0][0])

            t0 = RN152_Queue_Length_value * RN152ReferTime_value + PublishTime_ContourPic_value
            t1 = MNv2ReferTime1_value * MNv2_Queue_Length1_value
            t2 = MNv2ReferTime2_value * MNv2_Queue_Length2_value
            t3 = MNv2ReferTime3_value * MNv2_Queue_Length3_value
            #t_min = min(t0,t1,t2,t3)
            t_average = (t0 + t1 + t2 + t3)/4

            alpha_value = max(min(alpha_value - gamma1 * (t_average - query_interval), 0.9),0.6)
            #alpha_value = max(min(alpha_value - gamma1 * (t_min - query_interval), 0.9),0.5) 
            cursor.execute('update nodesdata set value=\'' + str(alpha_value) + '\' where name=\'alpha\'')
            print('alpha: ',alpha_value)

            cursor.close()
            conn.commit()
            conn.close()
        except:
            print('an error happen!')

MNv2ReferTime1 = []
MNv2ReferTime2 = []
MNv2ReferTime3 = []
RN152ReferTime = []
PublishTime_ContourPic = []
MNv2_Queue_Length1 = []
MNv2_Queue_Length2 = []
MNv2_Queue_Length3 = []
RN152_Queue_Length = []

client = mqtt.Client()
client.enable_logger(logger)
client.on_connect = on_connect
client.on_message = on_message
client.on_subscribe = on_subscribe
client.connect('172.17.0.6', 1883 , 60)

client.loop_forever()