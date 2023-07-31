import paho.mqtt.client as mqtt
import time
import os
from MobileNet_v2 import MobileNetV2
from keras.layers.core import Activation
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import cv2
import sqlite3
import pickle
from multiprocessing import Pool ,Manager
import logging 
import signal
logging.basicConfig(level=logging.WARNING) 
logger = logging.getLogger(__name__)

def handler(signum, frame):
    raise OSError("Timeout: 5s!")

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: "+str(rc))
    
#rc:  0: Connection successful 1: Connection refused - incorrect protocol version 2: Connection refused - invalid client identifier 3: Connection refused - server unavailable 4: Connection refused - bad username or password 5: Connection refused - not authorised 6-255: Currently unused.

def on_subscribe(client, userdata, mid, granted_qos): 
    print('subscribe ContourPic2MNv2 success')

def on_message(client, userdata, message):
    print('got one!')
    payload = message.payload
    if payload == b'stop':
        np.save('/home/edge/npy/time_frame_Eyedge.npy',np.array(time_frame))
        print('npy saved!')
    else:
        try:
            '''data decode'''
            payload_decoded = pickle.loads(payload)
            i = payload_decoded[0]
            videoname = payload_decoded[1]
            ContourPic = payload_decoded[2]
            Cnum = payload_decoded[3]
            #start_time_frame = payload_decoded[4]
            start_time_frame = time.time()
            payload = [i,videoname,ContourPic,Cnum,start_time_frame]

            queuePic.put(payload)
            publish('172.17.0.6',str(queuePic.qsize()),'data/MNv2_Queue_Length2',1883)
            #print('length of queuePic:',str(queuePic.qsize()))
        except:
            print('an error happen in receiving!')

def load_image(img,img_width,img_height):
    img = cv2.resize(img, (img_width, img_height))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor

def publish(broker,payload,topic,port):
    client = mqtt.Client()  
    client.enable_logger(logger)
    client.connect(broker, port , 60)
    client.loop_start()
    client.publish(topic,payload,2)
    client.loop_stop()
    client.disconnect()
    print('publish ' + topic +' success')

def loadMNv2(weightPath,alpha_MNv2):
    '''loading MobileNet_v2 model'''
    start_time_load = time.time()
    print('loading the MobileNet_v2 model...')

    classNum = 2
    img_width, img_height = 224,224

    base_model = MobileNetV2(input_shape=(img_width, img_height,3),
                    alpha=alpha_MNv2,
                    include_top=False,
                    weights= None
                    )

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) 
    x=Dense(1024,activation='relu')(x) 
    x=Dense(512,activation='relu')(x) 
    preds=Dense(classNum,activation='softmax')(x) #final layer with softmax activation
    model_MNv2=Model(inputs=base_model.input,outputs=preds)
    model_MNv2.load_weights(weightPath)
    print('done')
    end_time_load = time.time()
    print('loading time of MobileNet_v2 model: ' + str(end_time_load-start_time_load) + 's')
    return model_MNv2

def MNv2Refer(model_MNv2,payload,savePathPos,alpha):
    '''parameter'''
    flag = 0
    img_width, img_height =224,224

    '''ours'''
    # beta = (1 - alpha) * 0.3

    '''baseline'''
    # alpha = 0.7
    # beta = (1 - alpha) * 0.3
    
    '''all in edge'''
    alpha = 0.5     
    beta = 1 - alpha

    start_time_refer = time.time()

    i = payload[0]
    videoname = payload[1]
    ContourPic = payload[2]
    Cnum = payload[3]
    start_time_frame = payload[4]

    '''refer'''
    Image_preprocessed = load_image(ContourPic, img_width, img_height)
    pred = model_MNv2.predict(Image_preprocessed)
    MNv2ConfidenceValue = pred[0][0]

    #publish('172.17.0.7',str(MNv2ConfidenceValue),'data/sMNv2ConfidenceValue')

    if MNv2ConfidenceValue > alpha:
        #cv2.imwrite(savePathPos+ '/' + str(videoname) + "f" +str(i) + 'obj' +str(Cnum) +".jpg", ContourPic)
        print("got a query object!")
        print(str(videoname) + "f" +str(i)+ 'Obj' +str(Cnum))
        
    if MNv2ConfidenceValue <= alpha and MNv2ConfidenceValue > beta:
        flag = 1
        start_time_publish = time.time()

        payload = pickle.dumps(payload)

        publish('39.100.76.179',payload,'ContourPic2RN152',18833)
        end_time_publish = time.time()
        PublishTime_ContourPic = end_time_publish - start_time_publish
        publish('172.17.0.6',str(PublishTime_ContourPic),'data/PublishTime_ContourPic',1883)

    end_time_refer = time.time()
    MNv2ReferTime = end_time_refer - start_time_refer
    publish('172.17.0.6',str(MNv2ReferTime),'data/MNv2ReferTime2',1883)

    return start_time_frame,flag

def receiveMessage(queuePic,time_frame):
    client = mqtt.Client('MNv2Sub')
    client.enable_logger(logger)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_subscribe = on_subscribe

    client.connect('172.17.0.11', 1883 , 60)
    client.subscribe('ContourPic2MNv2',2)
    client.loop_forever()

def processMessage(queuePic,time_frame):
    weightPath = "/home/edge/weight.h5"
    savePathPos = '/home/edge/test'
    alpha_MNv2 = 0.5
    model_MNv2 = loadMNv2(weightPath,alpha_MNv2)
    signal.signal(signal.SIGALRM, handler)
    while True:
        if not queuePic.empty():
            try:   
                signal.alarm(5)
                payload = queuePic.get()
                publish('172.17.0.6',str(queuePic.qsize()),'data/MNv2_Queue_Length2',1883)
                try:
                    conn = sqlite3.connect('/home/edge/nodesdata.db')
                    cursor = conn.cursor()
                    cursor.execute('select * from nodesdata')
                    values = cursor.fetchall()
                    alpha_value = float(values[9][1])
                    cursor.close()
                    conn.close()
                except:
                    alpha_value = 0.8

                start_time_frame ,flag= MNv2Refer(model_MNv2,payload,savePathPos,alpha_value)
                publish('172.17.0.6',str(queuePic.qsize()),'data/MNv2_Queue_Length2',1883)

                if flag == 0:
                    time_frame.append(time.time()-float(start_time_frame))

                signal.alarm(0)
                
            except:
                continue


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    manager = Manager()
    queuePic = manager.Queue()
    time_frame = manager.list()
    p = Pool()
    pw = p.apply_async(receiveMessage,args=(queuePic,time_frame,)) 
    pr = p.apply_async(processMessage,args=(queuePic,time_frame,))
    p.close()
    p.join()
    