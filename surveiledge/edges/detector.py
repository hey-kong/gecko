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

def on_subscribe(client, userdata, mid, granted_qos): 
    print('subscribe Frame2Detector success')

def on_message(client, userdata, message):
    print('got one!')
    ObjectDetect(message.payload)

def on_publish(client, userdata, mid):
    print('ContourPic2Scheduler publish success!')

def ObjectDetect(payload):
    payload = pickle.loads(payload)
    
    i = payload[0]
    videoname = payload[1]
    frame1 = payload[2]
    frame2 = payload[3]
    frame3 = payload[4]
    start_time_frame = payload[5]

    print('got' + ' frame ' + str(i))

    frameDelta1 = cv2.absdiff(frame1, frame2)
    frameDelta2 = cv2.absdiff(frame2, frame3)

    thresh = cv2.bitwise_and(frameDelta1, frameDelta2) 
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=3)
    thresh = cv2.erode(thresh, None, iterations=1)
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Cnts = []

    for c in cnts:
        if cv2.contourArea(c) > 1024 * 4:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w+8) < (h+16)*2:
                Cnts.append([x,y,w,h])

    print('Cnts: ',Cnts)

    Cnum = 0

    for c in Cnts:  
        Cnum += 1                            
        x,y,w,h = c    
        cropImg = frame3[y-10 : y+h+10 , x-10 : x+w+10]
        if (h+20) * (w+20) > 300 * 300:
            cropImg = cv2.resize(cropImg, (224, 224))

        client = mqtt.Client()
        client.enable_logger(logger)
        client.on_publish = on_publish
        client.connect('127.0.0.1', 1883 , 60)
        client.loop_start()
        payload = [i,videoname,cropImg,Cnum,start_time_frame]
        payload = pickle.dumps(payload)
        client.publish('ContourPic2Scheduler' ,payload ,2)
        client.loop_stop()
        client.disconnect()
        
client = mqtt.Client()
client.enable_logger(logger)
client.on_connect = on_connect
client.on_message = on_message
client.on_subscribe = on_subscribe
client.connect('127.0.0.1', 1883 , 60)
client.subscribe('Frame2Detector',2)
client.loop_forever()