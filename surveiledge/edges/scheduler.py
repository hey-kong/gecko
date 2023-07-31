import paho.mqtt.client as mqtt
import time
import os
import numpy as np
import pickle
import sqlite3
import logging 
import pickle

# WarningPrint
logging.basicConfig(level=logging.WARNING) 
logger = logging.getLogger(__name__)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: "+str(rc))

def on_message(client, userdata, message):
    print('got one!')
    scheduler(message.payload)

def on_subscribe(client, userdata, mid, granted_qos): 
    print('subscribe ContourPic2Scheduler success')

def publish(broker,payload,topic,port):
    client = mqtt.Client()
    client.enable_logger(logger)
    client.connect(broker, port , 60)
    client.loop_start()
    client.publish(topic,payload,2)
    client.loop_stop()
    client.disconnect()
    print('publish ' + topic +' success')

def scheduler(payload):
    try:
        '''get input parameters'''
        conn = sqlite3.connect('/home/edge/parameters.db')
        cursor = conn.cursor()

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
        #alpha_value = float(values[9][1])

        cursor.close()
        conn.close()

        '''scheduling'''
        t0 = RN152_Queue_Length_value * RN152ReferTime_value + PublishTime_ContourPic_value
        t1 = MNv2ReferTime1_value * MNv2_Queue_Length1_value  
        t2 = MNv2ReferTime2_value * MNv2_Queue_Length2_value
        t3 = MNv2ReferTime3_value * MNv2_Queue_Length3_value
        
        if t1 >= t0 and t2 >= t0 and t3 >= t0:
            start_time_publish = time.time()
            publish('39.100.76.179',payload,'ContourPic2RN152',18833)
            end_time_publish = time.time()
            PublishTime_ContourPic = end_time_publish - start_time_publish
            publish('172.17.0.7',str(PublishTime_ContourPic),'data/PublishTime_ContourPic',1883)
        elif t2 >= t1 and t3 >= t1:
            publish('172.17.0.8',payload,'ContourPic2MNv2',1883)
            print('node1')
        elif t3 >= t2:
            publish('172.17.0.9',payload,'ContourPic2MNv2',1883) 
            print('node2')
        else:
            publish('172.17.0.10',payload,'ContourPic2MNv2',1883)
            print('node3')

        '''baseline or all in edge'''
        # publish('172.17.0.8',payload,'ContourPic2MNv2',1883)
        # publish('172.17.0.9',payload,'ContourPic2MNv2',1883)
        # publish('172.17.0.10',payload,'ContourPic2MNv2',1883)

        '''all in cloud'''
        # publish('39.100.76.179',payload,'ContourPic2RN152',18833)
    except:
        print('an error happen!')

client = mqtt.Client('SchedulerSub')
client.enable_logger(logger)
client.on_connect = on_connect
client.on_message = on_message
client.on_subscribe = on_subscribe
client.connect('127.0.0.1', 1883 , 60)
client.subscribe('ContourPic2Scheduler',2)
client.loop_forever()
