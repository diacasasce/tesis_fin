from time import sleep
import serial
from serial.tools.list_ports import *
import threading

def conectar():
    (serA,dA)=conectarA()
    (serB,dB)=conectarB()
    return (serA,dA,serB,dB)
def conectarA(to=-1):
    (devA,dA)=getA()
    if dA:
        if to<0:
            serA= serial.Serial(devA, 9600)
        else:
            serA= serial.Serial(devA, 9600,timeout=to)
        sleep(1)# Establish the connection on a specific port
        serA.flushInput()
        send_ser('-100;0;0;1;1;\r\n',serA)
        
    else:
        serA=dA
    return (serA,dA)
def conectarB():
    (devB,dB)=getB()
    if dB:
        serB= serial.Serial(devB, 9600) # Establish the connection on a specific port
    else:
        serB=dB
    return (serB,dB)
def send_ser(str,port,tmo=0):
    port.write(str.encode())
    rt=port.readline()
    port.flushInput()
    return rt
def send_nr(str,port):
    port.write(str.encode())
def getA():
    dA=False
    devA=False
    ports = list(comports())
    for p in ports:
        dev=list(grep("USB VID:PID=fff1:ff48"))
        for d in dev:
            dA=True
            print(d)
            devA=d[0]
    return (devA,dA)

def getB():
    dB=False
    devB=False
    ports = list(comports())
    for p in ports:
        dev=list(grep("USB VID:PID=2341:0042"))
        for d in dev:
            dB=True
            devB=d[0]
    return (devB,dB)

def on_B():
    (_,dB)=getB()
    return(dB)
def on_A():
    (_,dA)=getA()
    return(dA)


##for p in ports:
##    f=p.device
##    f=p.description
##    f=p.hwid