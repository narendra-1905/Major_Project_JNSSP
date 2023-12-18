import json
import socket 
from threading import Thread 
from socketserver import ThreadingMixIn

running = True

def startApplicationServer():
    class ClientThread(Thread): 
 
        def __init__(self,ip,port): 
            Thread.__init__(self) 
            self.ip = ip 
            self.port = port 
            print('Request received from Client IP : '+ip+' with port no : '+str(port)+"\n") 
 
        def run(self): 
            data = conn.recv(10000)
            data = json.loads(data.decode())
            start = int(str(data.get("start")))
            end = int(str(data.get("end")))
            output = ""
            for i in range(start,end):
                for j in range(1,10):
                    output+=str(i)+" * "+str(j)+" = "+str(i * j)+"\n"
            print("Edge cloud received request and sent output")        
            conn.send(output.encode())
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server.bind(('localhost', 2222))
    threads = []
    print("Multi Edge Cloud Started & waiting for incoming connections")
    while running:
        server.listen(4)
        (conn, (ip,port)) = server.accept()
        newthread = ClientThread(ip,port) 
        newthread.start() 
        threads.append(newthread) 
    for t in threads:
        t.join()

def startEdgeCloud():
    Thread(target=startApplicationServer).start()

startEdgeCloud()

