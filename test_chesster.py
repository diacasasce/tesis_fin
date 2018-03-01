
from chesster import Chesster
import serCOM as sC
import cv2
import numpy as np

##import train-data
player=Chesster(bright=70,sharp=100,contrast=100)

##inicio de juego
player.new_game()

##sC.send_ser("0;0;0;1;1;\r\n",player.arm)
##sC.send_ser("0;200;0;1;1;\r\n",player.arm)
##sC.send_ser("0;-200;0;1;1;\r\n",player.arm)
##sC.send_ser("0;0;0;1;1;\r\n",player.arm)
def run():
    print('start')
##    player.view(10)
##    im=player.captura('CP0.jpg')
    im=cv2.imread("CP0.jpg")
    player.plt_show(im)
    imc=player.calibracion(im)
    player.set_bright(70)
def play_alone():
    k=1
    d=[]
    t=[]
    while not(player.brain.is_over()):
        (d,t)=player.go_auto(True,na=str(k),d=d,t=t)
        k=k+1
##    if not(player.brain.is_over()):
##        turno =not(turno)
##        player.human()
        if input('break? ')!='':
            return (d,t)
    return (d,t)
def play_data(n,m):
    d=[]
    t=[]
    b=player.board
    for k in range(n,m):
        player.a_board=b
        (p,b,c,imc,d,t)=player.recon(na=str(k),CP=True,datt=d,tr=t)
        player.board=b
        print(k)
    return(d,t)
def get_data():
    d=[]
    
run()
    