import paho.mqtt.client as mqtt
import time
import numpy as np
import cv2
import pickle
import logging 

# WarningPrint
logging.basicConfig(level=logging.WARNING) 
logger = logging.getLogger(__name__)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: "+str(rc))
    video()
    client.disconnect()
    
def on_publish(client, userdata, mid):
    print('Frame2Detector publish success!')

def publish(broker,payload,topic,port):
    client = mqtt.Client()
    client.enable_logger(logger)
    client.connect(broker, port , 60)
    client.loop_start()
    client.publish(topic,payload,2)
    client.loop_stop()
    client.disconnect()
    print('publish ' + topic +' success')

def video():
    start_time_video = time.time()
    videoPath = '/home/camera/001004.avi'
    camera = cv2.VideoCapture(videoPath)
    num = int(camera.get(cv2.CAP_PROP_FPS))
    print('Video FPS: ',num)
    i=-1

    while camera.isOpened():
        i=i+1
        (ret, frame) = camera.read()

        if not ret:
            break

        if i%num == 1:
            frame1 =frame

        if i%num == 2:
            frame2 =frame

        if i%num == 3:
            frame3 =frame

            f = int(i/num)
            print(f)
            
            payload = [f,videoPath[-10:-4],frame1,frame2,frame3,time.time()]
            payload = pickle.dumps(payload)
            publish('172.17.0.2',payload,"Frame2Detector",1883)
            
    end_time_video = time.time()
    totalTime = end_time_video - start_time_video
    print('Total Time: ' + str(totalTime) + 's')
    print('FramesPerSecond: ' + str(f/totalTime) + '/s')

client = mqtt.Client('CameraConnect')
client.enable_logger(logger)
client.on_connect = on_connect
client.connect('172.17.0.2', 1883 , 60)
client.loop_forever()
