import matplotlib.pyplot as plt
import numpy
import scipy
#from scipy import signal
a = 0
b = 10
n = 500
Fs = 1000
F_max = 23
#random = numpy.random.normal(a, b, n)
time = numpy.arange(n)/Fs
#x = time
w = F_max/(Fs/2)
#parameters_filter = scipy.signal.butter(3, w, 'low', output='sos')
#y = scipy.signal.sosfiltfilt(parameters_filter, random)
#fig, ax = plt.subplots(figsize=(21,14))
#ax.plot(x, y, linewidth=1)
#ax.set_xlabel("Час(секунди)", fontsize=14)
#ax.set_ylabel("Амплітуда сигналу", fontsize=14)
#plt.title("Сигнал з максимальною частотою F_max = 23Гц", fontsize=14)
#plt.show()
#from scipy import signal, fft
random = numpy.random.normal(a, b, n)
parameters_filter = scipy.signal.butter(3, w, 'low', output='sos')
filtered_signal = scipy.signal.sosfiltfilt(parameters_filter, random)
spectrum = scipy.fft.fft(filtered_signal)
numpy.abs(scipy.fft.fftshift(spectrum))
frequency = scipy.fft.fftfreq(500, 1/500)
filtered_signal = scipy.fft.fftshift(frequency)
fig, ax = plt.subplots(figsize=(21,14))
y = spectrum
x = frequency
ax.plot(x, y, linewidth=1)
ax.set_xlabel("Час(секунди)", fontsize=14)
ax.set_ylabel("Амплітуда сигналу", fontsize=14)
plt.title("Сигнал з максимальною частотою F_max = 23Гц", fontsize=14)
plt.show()