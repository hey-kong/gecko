import paho.mqtt.client as mqtt
import time
import numpy as np
import os
from model import ResNet152
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
import pickle
import cv2
from multiprocessing import Pool ,Manager
import logging 
logging.basicConfig(level=logging.WARNING) 
logger = logging.getLogger(__name__)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: "+str(rc))

def on_subscribe(client, userdata, mid, granted_qos): 
    print('subscribe ContourPic2ResNet successful')

def on_message(client, userdata, message):
    print('got one!')
    payload = message.payload
    if payload == b'stop':
        np.save('/home/cloud/npy/time_frame_cloud_baseline3.npy',np.array(time_frame))
        print('npy saved!')
    else:
        queuePic.put(payload)
        publish('172.17.0.2',str(queuePic.qsize()),'data/RN152_Queue_Length')

def publish(broker,payload,topic):
    #RN152Pub
    client = mqtt.Client()
    client.enable_logger(logger)
    client.connect(broker, 1883 , 60)
    client.loop_start()
    client.publish(topic,payload,2)
    client.loop_stop()
    client.disconnect()
    print('publish ' + topic +' success')

def load_image(img,img_width,img_height):
    x = cv2.resize(img,(img_width,img_height))
    x = preprocess_input(x)
    if x.ndim == 3:
        x = np.expand_dims(x, 0)
    return x

def loadRN152():
    #loading ResNet152 model
    start_time_RN152 = time.time()
    print('loading the ResNet152 model...')
    model_RN152 = ResNet152()
    print('done')
    print('loading time of ResNet152 model: ' + str(time.time()-start_time_RN152) + 's')
    return model_RN152

def RN152Refer(model_RN152,payload,savePathPos,queryClass):
    '''parameter'''
    img_width, img_height =224,224

    start_time_refer = time.time()

    '''data decode'''
    payload_decoded = pickle.loads(payload)
    i = payload_decoded[0]
    videoname = payload_decoded[1]
    ContourPic = payload_decoded[2]
    Cnum = payload_decoded[3]
    start_time_frame = payload_decoded[4]

    '''refer'''
    Image_preprocessed = load_image(ContourPic,img_width,img_height)
    y = model_RN152.predict(Image_preprocessed)
    pred_title = decode_predictions(y, top=1)[0][0][1]
    print(pred_title)

    end_time_refer = time.time()
    RN152ReferTime = end_time_refer - start_time_refer
    publish('172.17.0.2',str(RN152ReferTime),'data/RN152ReferTime')

    if pred_title == queryClass:
        #cv2.imwrite(savePathPos+ '/' + str(videoname) + "f" +str(i) + 'obj' +str(Cnum) +".jpg", ContourPic)
        print("got a query object!")
        print(str(videoname) + "f" +str(i)+ 'Obj' +str(Cnum))
    
    return start_time_frame

def receiveMessage(queuePic,time_frame):
    client = mqtt.Client()
    client.enable_logger(logger)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_subscribe = on_subscribe

    client.connect('172.17.0.2', 1883 , 60)
    client.subscribe('ContourPic2RN152',2)
    client.loop_forever()

def processMessage(queuePic,time_frame):
    queryClass = 'moped'
    savePathPos = '/home/cloud/test'
    model_RN152 = loadRN152()
    while True:
        if not queuePic.empty():
            payload = queuePic.get()
            try:           
                start_time_frame = RN152Refer(model_RN152,payload,savePathPos,queryClass)
                time_frame.append(time.time()-float(start_time_frame))
                publish('172.17.0.2',str(queuePic.qsize()),'data/RN152_Queue_Length')
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
    

