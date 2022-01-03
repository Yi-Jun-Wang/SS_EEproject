import matplotlib.pyplot as plt
import numpy as np
import time
import random
import math
import serial
from collections import deque
from scipy.signal import filtfilt, butter, lfilter
from tensorflow.keras.models import load_model
#import thingspeak
#Display loading 
class PlotData:
    def __init__(self, max_entries=30):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)
        
model = load_model('C:/Users/User/Desktop/SS_exp_project-main/CrestDetectRnn_Best.h5')

#initial
fig, (ax,ax2) = plt.subplots(2,1)
line,  = ax.plot(np.random.randn(100))
line2, = ax2.plot(np.random.randn(100))
plt.show(block = False)
plt.setp(line2, color = 'r')

fig2, (ax3,ax4) = plt.subplots(2,1)
line3, = ax3.plot(np.random.randn(100))
line4, = ax4.plot(np.random.randn(100))
plt.show(block = False)

fig3, (ax5,ax6) = plt.subplots(2,1)
line5, = ax5.plot(np.random.randn(100))
line6, = ax6.plot(np.random.randn(100))
plt.show(block = False)

fig4, (ax7,ax8) = plt.subplots(2,1)
line7, = ax7.plot(np.random.randn(100))
#line8, = ax8.plot(np.random.randn(100))
plt.show(block = False)

fig5 = plt.figure(figsize=(8, 8))
ax8 = plt.subplot()
ax8.set_xlim((-2, 2))
ax8.set_ylim((-2, 2))
plt.show(block = False)

fig6 = plt.figure(figsize=(8, 8))
ax9 = plt.subplot()
ax9.set_xlim((-2, 2))
ax9.set_ylim((-2, 2))
plt.show(block = False)

angle = np.linspace(-np.pi, np.pi, 50)
cirx = np.sin(angle)
ciry = np.cos(angle)
ax8.plot(cirx, ciry, 'k-')
ax9.plot(cirx, ciry, 'k-')

PData= PlotData(500)
PData_AC = PlotData(500)

ax.set_ylim(0, 500)
ax2.set_ylim(-5, 5)
ax3.set_ylim(0, 1000)
ax4.set_ylim(0, 500)
ax5.set_ylim(-5, 5)
ax6.set_ylim(-5, 5)
ax7.set_ylim(0, 1)
# plot parameters
print ('plotting data...')
# open serial port
strPort='com4'
ser = serial.Serial(strPort, 115200)
ser.flush()
#timestep = 18.0/115200
timestep = 0.01
#FIR = scipy.signal.freqz_zpk([np.exp(1j*np.pi), np.exp(0.9j*np.pi), np.exp(0.8j*np.pi), np.exp(-1j*np.pi), np.exp(-0.9j*np.pi), np.exp(-0.8j*np.pi)], [0, 0, 0, 0, 0, 0], 1)
FIR = np.ones(20, dtype=np.float32)/20.0
b, a = butter(5, 0.1, 'lowpass')
z, p, k = butter(5, 0.1, 'lowpass', output='zpk')
ax8.plot(np.real(z), np.imag(z), 'o', markersize=4)
ax8.plot(np.real(p), np.imag(p), 'x', markersize=4)
z = np.roots(FIR)
ax9.plot(np.real(z), np.imag(z), 'o', markersize=4)
ax9.plot(0, 0, 'x', markersize=4)

CrestStep = deque(maxlen = 10)
CrestStep.append(0)
HeartRate = 0
num_traindata = 0
datain_count = 0
#IoT = thingspeak.Channel(1270269, api_key='5B8HVGP6DFGMVRU0')
start = time.time()
while True:
    
    for ii in range(10):

        try:
            data = float(ser.readline())
            PData.add(time.time() - start, data)        
            dataset = list(PData.axis_y)
            avg = np.average(dataset)
            data = data - avg
            PData_AC.add(0, data)
        except:
            pass
    #Signalupload = {'field1':data}
    #IoT.update(Signalupload)
    datain_count += 1
    SerialData = list(PData_AC.axis_y)
    SerialData_fft = np.fft.fft(SerialData)
    SerialData_fft_FilterDc = SerialData_fft
    SerialData_fft_FilterDc[0] = 0
    PData_avgfiltered = list(lfilter(FIR, 1, PData_AC.axis_y))
    
    try:
        PData_filtered = filtfilt(b, a, PData_AC.axis_y)
    except:
        PData_filtered = list(PData_AC.axis_y)
    
    #將輸入丟進model預測
    try:
        X = np.array(PData_filtered[-10:])
        X = X.reshape((1, 10, 1))
        Crestexist = np.argmax(model.predict(X), axis=-1)
        print(Crestexist)
        if(Crestexist[0]):
            x = list(PData.axis_x)
            Timeaxis = np.array(x[-10:])
            CrestTime = np.mean(Timeaxis)
            difftime = CrestTime - CrestStep[-1]
            if(difftime > 0.4):
                CrestStep.append(CrestTime)
                HeartRate = np.mean(np.diff(list(CrestStep)))
                HeartRate = 60.0/HeartRate            
    except:
        pass
    
    '''
    #製作dataset
    try:
        if(datain_count%10 == 0):
            input_name = 'D:/GitHub/SS_exp_project/trainset/signal'+str(num_traindata)+'.txt'
            input_name2 = 'D:/GitHub/SS_exp_project/trainset/signal'+str(num_traindata)+'_1.txt'
            label_name = 'D:/GitHub/SS_exp_project/trainset/label'+str(num_traindata)+'.txt'
            label_name2 = 'D:/GitHub/SS_exp_project/trainset/label'+str(num_traindata)+'_1.txt'
            signal = np.array(PData_filtered[-10:])
            if(np.max(signal)>2):
                label = np.array([1])
            else:
                label = np.array([0])
            np.savetxt(input_name, signal)
            np.savetxt(label_name, label)
            signal = signal*2
            np.savetxt(input_name2, signal)
            np.savetxt(label_name2, label)
            num_traindata += 1
    except:
        pass
    '''

    '''
    #利用Threshold測量心率
    try:
        x = list(PData.axis_x)
        Timeaxis = np.array(x[-20:])
        Datanew = np.array(PData_filtered[-20:])
        diff = np.max(Datanew)-np.min(Datanew)
        if diff > 1.35:
            CrestTime = np.mean(Timeaxis)
            difftime = CrestTime - CrestStep[-1]
            if(difftime > 0.4):
                CrestStep.append(CrestTime)
            HeartRate = np.mean(np.diff(list(CrestStep)))
            HeartRate = 60.0/HeartRate
        #HeartRateUpload = {'field2':HeartRate}
        #IoT.update(HeartRateUpload)
        #print(HeartRate)
    except:
        pass
    '''
    f = np.fft.fftfreq(len(SerialData_fft), d=timestep)
    f_min = np.min(f)
    f_max = np.max(f)
    Serialabs = np.absolute(SerialData_fft_FilterDc)
    index = np.argmax(Serialabs)
    Mainfreq = abs(f[index])
    Amp = Serialabs[index]
    ax.set_xlim(PData.axis_x[0], PData.axis_x[0]+5)
    ax2.set_xlim(PData.axis_x[0], PData.axis_x[0]+5)
    ax3.set_xlim(f_min, f_max)
    ax4.set_xlim(0, 50)
    ax5.set_xlim(PData.axis_x[0], PData.axis_x[0]+5)
    ax6.set_xlim(PData.axis_x[0], PData.axis_x[0]+5)
    ax7.set_xlim(0, 150)
    line.set_xdata(PData.axis_x)
    line.set_ydata(PData.axis_y)
    line2.set_xdata(PData.axis_x)
    line2.set_ydata(PData_AC.axis_y)
    line3.set_xdata(f)
    line3.set_ydata(abs(SerialData_fft))
    line4.set_xdata((Mainfreq, Mainfreq))
    line4.set_ydata((0, Amp))
    line5.set_xdata(PData.axis_x)
    line5.set_ydata(PData_avgfiltered)
    line6.set_xdata(PData.axis_x)
    line6.set_ydata(PData_filtered)
    line7.set_xdata((HeartRate, HeartRate))
    line7.set_ydata((0, 1))
    #fig.canvas.draw()
    #fig.canvas.flush_events()
    #fig2.canvas.draw()
    #fig2.canvas.flush_events()
    fig3.canvas.draw()
    fig3.canvas.flush_events()
    fig4.canvas.draw()
    fig4.canvas.flush_events()