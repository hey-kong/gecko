import paho.mqtt.client as mqtt
from makeQueryTrainData1 import makeQueryTrainData 
from queryTrain1 import queryTrain 
import hashlib
import time
import shutil
import numpy as np
import os
from sys import argv
from model import ResNet152
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: "+str(rc))

def on_publish(client, userdata, mid):
    #logging.debug("pub ack "+ str(mid))
    client.mid_value=mid
    client.puback_flag=True  

def on_subscribe(client, userdata, mid, granted_qos):
    print('subscribe ResNet successful')
    global num
    num = 0


def on_message(client, userdata, message):
    print("got a picture")
    global num
    num += 1
    print(num)

    cropImg = np.array(eval(message.payload))
    Image_preprocessed = preprocess(cropImg)
    y = model_RN152.predict(Image_preprocessed)
    pred_title = decode_predictions(y, top=1)[0][0][1]
    print(pred_title)
    if pred_title == queryClass:
        j += 1
        cv2.imwrite(savePath+ '/' + "f" +str(j) + ".jpg", cropImg)
        print('picture saved!')
        

## waitfor loop
def wait_for(client,msgType,period=0.25,wait_time=40,running_loop=False):
    client.running_loop=running_loop #if using external loop
    wcount=0  
    while True:
        #print("waiting"+ msgType)
        if msgType=="PUBACK":
            if client.on_publish:        
                if client.puback_flag:
                    return True
     
        if not client.running_loop:
            client.loop(.01)  #check for messages manually
        time.sleep(period)
        #print("loop flag ",client.running_loop)
        wcount+=1
        if wcount>wait_time:
            print("return from wait loop taken too long")
            return False
    return True 


def send_header(filename,client,topic,qos):
   header="header"+",,"+filename+",,"
   header=bytearray(header,"utf-8")
   header.extend(b','*(200-len(header)))
   print(header)
   c_publish(client,topic,header,qos)


def send_end(filename,client,topic,qos,out_hash_md5):
   end="end"+",,"+filename+",,"+out_hash_md5.hexdigest()
   end=bytearray(end,"utf-8")
   end.extend(b','*(200-len(end)))
   print(end)
   c_publish(client,topic,end,qos)


def c_publish(client,topic,out_message,qos):
   res,mid=client.publish(topic,out_message,qos)  #publish
   if res==0: #published ok
      if wait_for(client,"PUBACK",running_loop=True):
         if mid==client.mid_value:
            print("match mid ",str(mid))
            client.puback_flag=False  #reset flag
         else:
            raise SystemExit("not got correct puback mid so quitting")
         
      else:
         raise SystemExit("not got puback so quitting")


def send(filename,client,topic1,qos):
    fo=open(filename,"rb+")
    print(filename + ' is opened')
    send_header(filename,client,topic1,qos)
    Run_flag=True
    out_hash_md5 = hashlib.md5()
    while Run_flag:
        chunk=fo.read(data_block_size)
        if chunk:
            out_hash_md5.update(chunk) #update hash
            out_message=chunk
            #print(" length =",type(out_message))
            c_publish(client,topic1,out_message,qos)
                
        else:
            #end of file so send hash
            out_message=out_hash_md5.hexdigest()
            send_end(filename,client,topic1,qos,out_hash_md5)
            #print("out Message ",out_message)
            #res,mid=client.publish(topic1,out_message,qos)
            Run_flag=False
            
    fo.close()


def preprocess(x):
    x = cv2.resize(x,(224,224))
    #x = resize(x, (224,224), mode='constant') * 255
    x = preprocess_input(x)
    if x.ndim == 3:
        x = np.expand_dims(x, 0)
    return x


def delete(Path):
    if os.path.exists(Path):
        os.remove(Path)
        print(Path + ' deleted')
    else:
        print(Path + ' not exist')


queryClass = argv[1]
broker = "172.17.0.3"
topic1="weight1"
alpha = 0.5

qos=2
data_block_size=1000000000

makeQueryTrainData(queryClass)
weightPath1, weightPath2 = queryTrain(queryClass,alpha)

client = mqtt.Client('cloud')
client.on_connect = on_connect
client.on_publish=on_publish
client.puback_flag=False   #use flag in publish ack
client.mid_value=None
client.connect(broker, 1883, 60)
client.loop_start() 

start=time.time()
print("publishing ")
send(weightPath1,client,topic1,qos)

time_taken=time.time()-start
print("took time of Publish: ",time_taken)
# delete(weightPath1)
# delete(weightPath2)

time.sleep(3)
client.disconnect()   
client.loop_stop()  


###################
print("ResNet Client Connecting...")
client = mqtt.Client('ResNet')
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message

# savePath = '/home/cloud/test'
# if not os.path.exists(savePath):
#     os.makedirs(savePath)

client.connect(broker, 1883, 60)
client.subscribe('ResNet',qos=2)

# #loading ResNet152 model
# start_time_RN152=time.time()
# print('loading the ResNet152 model...')
# j = 0
# model_RN152 = ResNet152()
# print('done')
# print('loading time of ResNet152 model: ' + str(time.time()-start_time_RN152) + 's')

client.loop_forever()