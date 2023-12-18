import socket
import time
import sys
import tkinter
from tkinter import *
import math
import random
from threading import Thread 
from collections import defaultdict
from tkinter import ttk
from tkinter import filedialog
from multiprocessing import Queue
import matplotlib.pyplot as plt
import cv2
import socket
import struct
import time
import pickle
import zlib
import numpy as np
from PIL import Image
import json

global mobile
global labels
global mobile_x
global mobile_y
global text, text1
global canvas
global mobile_list
global filename

global edge, edge_label, edge_x, edge_y

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

queue = Queue()
click = False
shownOutput = False
files = []
global local_time
global offload_time, HHO_time, hho_uneven, propose_throughput, propose_latency, extension_throughput, extension_latency
global extension_time
global load1
global load2
global load3
load1 = 0
load2 = 0
load3 = 0
existing_time = 0
propose_time = 0
total_request = 0
hho_uneven = 0
propose_throughput = 0
propose_latency = 0
extension_throughput = 0
extension_latency = 0

def calculateDistance(iot_x,iot_y,x1,y1):
    flag = False
    for i in range(len(iot_x)):
        dist = math.sqrt((iot_x[i] - x1)**2 + (iot_y[i] - y1)**2)
        if dist < 60:
            flag = True
            break
    return flag


def startMobileSimulation(mobile_x,mobile_y,canvas,text,mobile,labels):
    class SimulationThread(Thread):
        def __init__(self,mobile_x,mobile_y,canvas,text,mobile,labels): 
            Thread.__init__(self) 
            self.mobile_x = mobile_x
            self.mobile_y = mobile_y
            self.canvas = canvas
            self.text = text
            self.mobile = mobile
            self.labels = labels
 
        def run(self):
            while(True):
                for i in range(len(mobile_x)):
                    x = random.randint(80, 500)
                    y = random.randint(50, 600)
                    flag = calculateDistance(mobile_x,mobile_y,x,y)
                    if flag == False:
                        mobile_x[i] = x
                        mobile_y[i] = y
                        canvas.delete(mobile[i])
                        canvas.delete(labels[i])
                        name = canvas.create_oval(x,y,x+40,y+40, fill="blue")
                        lbl = canvas.create_text(x+20,y-10,fill="darkblue",font="Times 10 italic bold",text="Mobile "+str(i))
                        labels[i] = lbl
                        mobile[i] = name
                canvas.update()
                PSOJNSSP()
                if click == True and filename not in files:
                    print(filename)
                    #queue.put(filename)
                    files.append(filename)
                    
                time.sleep(4)
                   
                    
    newthread = SimulationThread(mobile_x,mobile_y,canvas,text,mobile,labels) 
    newthread.start()
    
    
def generate():
    global mobile
    global labels
    global mobile_x
    global mobile_y
    global edge, edge_label, edge_x, edge_y
    edge = []
    edge_label = []
    edge_x = []
    edge_y = []
    mobile = []
    mobile_x = []
    mobile_y = []
    labels = []

    edge_x.append(10)
    edge_y.append(100)
    edge_x.append(10)
    edge_y.append(300)
    edge_x.append(10)
    edge_y.append(500)
    name = canvas.create_oval(10,100,45,140, fill="green")
    lbl = canvas.create_text(30,90,fill="green",font="Times 10 italic bold",text="Edge 1")
    edge.append(name)
    edge_label.append(lbl)

    name = canvas.create_oval(10,300,45,340, fill="green")
    lbl = canvas.create_text(30,290,fill="green",font="Times 10 italic bold",text="Edge 2")
    edge.append(name)
    edge_label.append(lbl)

    name = canvas.create_oval(10,500,45,540, fill="green")
    lbl = canvas.create_text(30,490,fill="green",font="Times 10 italic bold",text="Edge 3")
    edge.append(name)
    edge_label.append(lbl)

    for i in range(0,20):
        run = True
        while run == True:
            x = random.randint(80, 500)
            y = random.randint(50, 600)
            flag = calculateDistance(mobile_x,mobile_y,x,y)
            if flag == False:
                mobile_x.append(x)
                mobile_y.append(y)
                run = False
                name = canvas.create_oval(x,y,x+40,y+40, fill="blue")
                lbl = canvas.create_text(x+20,y-10,fill="darkblue",font="Times 10 italic bold",text="Mobile "+str(i))
                labels.append(lbl)
                mobile.append(name)
    startMobileSimulation(mobile_x,mobile_y,canvas,text,mobile,labels)

def offloadThread():
    global shownOutput
    global offload_time, propose_latency, propose_throughput
    class OffloadThread(Thread):
        def __init__(self):
            Thread.__init__(self)
                        
        def run(self):
            global shownOutput
            global offload_time,propose_latency, propose_throughput
            while not queue.empty():
                if shownOutput:
                    print(str(shownOutput))
                    filename = queue.get()
                    img = cv2.imread(filename)
                    result, img = cv2.imencode('.jpg', img, encode_param)
                    mid = int(mobile_list.get())
                    worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    worker.connect(('localhost', 8485))
                    data = pickle.dumps(img, 0)
                    size = len(data)
                    worker.sendall(struct.pack(">L", size) + data)
                    data = b""
                    payload_size = struct.calcsize(">L")
                    while len(data) < payload_size:
                        #print("Recv: {}".format(len(data)))
                        data += worker.recv(4096)
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack(">L", packed_msg_size)[0]
                    print("msg_size: {}".format(msg_size))
                    while len(data) < msg_size:
                        start = time.time() - 0.02
                        data += worker.recv(4096)
                    frame_data = data[:msg_size]
                    data = data[msg_size:]
                    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    worker.close()
                    end = time.time()
                    click = False
                    files.clear()
                    offload_time = end - start
                    propose_latency = 8.0 * offload_time
                    propose_throughput = (4096/1000) / (offload_time * 8.0)
                    text1.insert(END,"Mobile Task Offloading Computation Cost : "+str(offload_time)+"\n")
                    text1.insert(END,"Propose Offloading Latency : "+str(propose_latency)+"\n")
                    text1.insert(END,"Propose Offloading Throughput : "+str(propose_throughput)+"\n")
                    print("Mobile Task Offloading Computation Cost : "+str(offload_time))
                    shownOutput = False
                    cv2.imshow("Detected Faces ", frame)
                    cv2.waitKey(0)
    if shownOutput:
        newthread = OffloadThread() 
        newthread.start()            

def HHOoffloadTask():
    global filename
    global click
    global shownOutput
    files.clear()
    queue.put(filename)
    click = True
    shownOutput = True
    runHHO()

def offloadTask():
    global filename
    global click
    global shownOutput
    files.clear()
    filename = filedialog.askopenfilename(initialdir="images")
    queue.put(filename)
    click = True
    shownOutput = True
    PSOJNSSP()

def runLocalOffloadSimulation():
    global total_request
    global existing_time
    global propose_time
    local_time_start = time.time()
    output=''
    for i in range(4,8):
        for j in range(1,20):
            output+=str(i)+" * "+str(j)+" = "+str(i * j)+"\n"
    local_time_end = time.time()
    existing_time = existing_time + (local_time_end - local_time_start)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 2222))
    jsondata = json.dumps({"start": "4", "end": "8"})
    offload_time_start = time.time()
    message = client.send(jsondata.encode())
    offload_time_end = time.time()
    propose_time = propose_time + (offload_time_end - offload_time_start)
    total_request = total_request + 1

#initializing all horses and here we are considering edge servers as the horses and the horse with best fitness will be conisder as dominant horse
#and this horse will be selected for service placement or processing which means edge server with best fitness or dominance will be selecetd    
def initHorsePopulation(source_node):
    horses = []
    for i in range(len(edge_x)): 
        if i != source_node:
            horses.append([edge_x[i], edge_y[i]])
    return horses        
#here each edge server fitness will be evaluated and the edge server with least distnace will be selected for service placement     
def evaluateFitness(horses, x, y):
    fitness = 0
    dominant_horse = 0
    best_fit = 300
    for m in range(len(horses)):
        fit = math.sqrt((x - horses[m][0])**2 + (y - horses[m][1])**2)
        if fit < best_fit:
            best_fit = fit
            dominant_horse = m
    return dominant_horse    
        

def HHOoffloadThread():
    global shownOutput
    global HHO_time, extension_latency, extension_throughput, hho_uneven
    class HHOOffloadThread(Thread):
        def __init__(self):
            Thread.__init__(self)
                        
        def run(self):
            global shownOutput
            global HHO_time, hho_uneven, extension_latency, extension_throughput
            while not queue.empty():
                if shownOutput:
                    print(str(shownOutput))
                    filename = queue.get()
                    img = cv2.imread(filename)
                    result, img = cv2.imencode('.jpg', img, encode_param)
                    mid = int(mobile_list.get())
                    worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    worker.connect(('localhost', 8485))
                    data = pickle.dumps(img, 0)
                    size = len(data)
                    worker.sendall(struct.pack(">L", size) + data)
                    data = b""
                    payload_size = struct.calcsize(">L")
                    while len(data) < payload_size:
                        #print("Recv: {}".format(len(data)))
                        data += worker.recv(4096)
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack(">L", packed_msg_size)[0]
                    print("msg_size: {}".format(msg_size))
                    start = time.time()
                    while len(data) < msg_size:
                        data += worker.recv(4096)
                        start = time.time()
                    frame_data = data[:msg_size]
                    data = data[msg_size:]
                    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    worker.close()
                    end = time.time()
                    click = False
                    files.clear()
                    HHO_time = end - start
                    extension_throughput = (4096/1000) / (HHO_time * 8.0)
                    extension_latency = 8.0 * HHO_time
                    hho_uneven = (hho_uneven + HHO_time)/1000
                    text1.insert(END,"HHO Mobile Task Offloading Computation Cost : "+str(HHO_time)+"\n")
                    text1.insert(END,"HHO Mobile Task Offloading Latency : "+str(extension_latency)+"\n")
                    text1.insert(END,"HHO Mobile Task Offloading Throughput : "+str(extension_throughput)+"\n")
                    print("HHO Mobile Task Offloading Computation Cost : "+str(HHO_time))
                    shownOutput = False
                    cv2.imshow("Detected Faces ", frame)
                    cv2.waitKey(0)
    if shownOutput:
        newthread = HHOOffloadThread() 
        newthread.start() 

def runHHO():
    global shownOutput, click
    global HHO_time
    global load1, load2, load3
    text.delete('1.0', END)
    offload_1 = 0
    offload_2 = 0
    selected = int(mobile_list.get())
    x = mobile_x[selected]
    y = mobile_y[selected]
    canvas.delete(labels[selected])
    lbl = canvas.create_text(x+20,y-10,fill="red",font="Times 10 italic bold",text="Mobile "+str(selected))
    labels[selected] = lbl                
    horses = initHorsePopulation(selected)
    dominant_horse = evaluateFitness(horses, x, y)
    if dominant_horse == 0:
        load1+=1
    if dominant_horse == 1:
        load2+=1
    if dominant_horse == 2:
        load3+=1
    text.insert(END,"\n\nCurrent Selected Mobile "+str(selected)+" offloading task to HHO optimized EDGE "+str(dominant_horse+1))            
    if click:
        HHOoffloadThread()
        click = False
    
def PSOJNSSP():
    global shownOutput, click
    global offload_time
    global load1, load2, load3
    text.delete('1.0', END)
    offload_1 = 0
    offload_2 = 0
    selected = int(mobile_list.get())
    x = mobile_x[selected]
    y = mobile_y[selected]
    canvas.delete(labels[selected])
    lbl = canvas.create_text(x+20,y-10,fill="red",font="Times 10 italic bold",text="Mobile "+str(selected))
    labels[selected] = lbl                
    distance = 300
    for i in range(len(edge_x)): #loop all edge particles and selected user
        if i != selected:
            x1 = edge_x[i]
            y1 = edge_y[i]
            dist = math.sqrt((x - x1)**2 + (y - y1)**2) #search the space for best solution by using distnace and queue size
            if dist < distance: #if minimum distnace 
                offload_2 = offload_1 
                offload_1 = i #choose as best optimal solution
                distance = dist
                text.insert(END,"Selected Mobile "+str(selected)+" is Nearer to Edge "+str(i+1)+"\n")
    runLocalOffloadSimulation()
    if offload_1 == 0:
        load1+=1
    if offload_1 == 1:
        load2+=1
    if offload_1 == 2:
        load3+=1
    text.insert(END,"\n\nCurrent Selected Mobile "+str(selected)+" offloading task to EDGE "+str(offload_1+1))            
    if click:
        offloadThread()
        click = False


def localRun():
    global local_time
    text1.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="images")
    local_time = time.time()
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    frame = cv2.imread(filename)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    local_time = time.time() - local_time
    text1.insert(END,"Local Running Computation Cost : "+str(local_time)+"\n")
    print("Local Running Computation Cost : "+str(local_time))
    cv2.imshow("Num Faces found ", frame)
    cv2.waitKey(0)

def graph():
    print(str(local_time)+" "+str(offload_time))
    height = [local_time,offload_time,HHO_time]
    bars = ('Evenly Local', 'Evenly Offload','Evenly HHO Optimized')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Tasks Evenly Distributed Computational Cost Graph")
    plt.show()

def unevengraph():
    print(str(existing_time)+" uneven "+str(propose_time))
    height = [existing_time,propose_time,hho_uneven]
    bars = ('Unevenly Local', 'Unevenly Offload','Unevenly HHO Optimized')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Num Task "+str(total_request)+" Computational Cost when tasks evenly distributed")
    plt.show()


def latencyGraph():
    height = [propose_latency,extension_latency]
    bars = ('Propose Latency', 'HHO Latency')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Latency Graph")
    plt.show()

def throughputGraph():
    height = [propose_throughput,extension_throughput]
    bars = ('Propose Throughput', 'HHO Throughput')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Throughput Graph")
    plt.show()    

def Main():
    global text
    global text1
    global canvas
    global mobile_list
    root = tkinter.Tk()
    root.geometry("1300x1200")
    root.title("Joint Network Selection and Service Placement Based on Particle Swarm Optimization for Multi-Access Edge Computing")
    root.resizable(True,True)
    font1 = ('times', 12, 'bold')

    canvas = Canvas(root, width = 1200, height = 700)
    canvas.pack()

    l1 = Label(root, text='Mobile ID:')
    l1.config(font=font1)
    l1.place(x=850,y=10)

    mid = []
    for i in range(0,20):
        mid.append(str(i))
    mobile_list = ttk.Combobox(root,values=mid,postcommand=lambda: mobile_list.configure(values=mid))
    mobile_list.place(x=1000,y=10)
    mobile_list.current(0)
    mobile_list.config(font=font1)

    createButton = Button(root, text="Generate Mobile Devices", command=generate)
    createButton.place(x=850,y=60)
    createButton.config(font=font1)

    localButton = Button(root, text="Local Run Task", command=localRun)
    localButton.place(x=850,y=110)
    localButton.config(font=font1)

    offloadButton = Button(root, text="Offload Evenly Task", command=offloadTask)
    offloadButton.place(x=1000,y=110)
    offloadButton.config(font=font1)

    HHOButton = Button(root, text="HHO Optimized Offload Evenly Task", command=HHOoffloadTask)
    HHOButton.place(x=850,y=160)
    HHOButton.config(font=font1)

    graphButton = Button(root, text="Unevenly Task Computation Cost Graph", command=unevengraph)
    graphButton.place(x=850,y=210)
    graphButton.config(font=font1)

    unevengraphButton = Button(root, text="Evenly Task Computation Cost Graph", command=graph)
    unevengraphButton.place(x=850,y=250)
    unevengraphButton.config(font=font1)

    latencyButton = Button(root, text="Latency Graph", command=latencyGraph)
    latencyButton.place(x=850,y=300)
    latencyButton.config(font=font1)

    throughputButton = Button(root, text="Throughput Graph", command=throughputGraph)
    throughputButton.place(x=1000,y=300)
    throughputButton.config(font=font1)

    text=Text(root,height=12,width=60)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=750,y=350)

    text1=Text(root,height=10,width=70)
    scroll1=Scrollbar(text1)
    text1.configure(yscrollcommand=scroll1.set)
    text1.place(x=750,y=520)
    
    
    root.mainloop()
   
 
if __name__== '__main__' :
    Main ()
    
