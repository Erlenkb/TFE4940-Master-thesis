### SCATERING measurments


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import values


####### Font values #######
SMALL_SIZE = 15
MEDIUM_SIZE = 17
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
###########################


######## Third octave settings ########
x_ticks_third_octave = [100, 200, 500, 1000, 2000, 5000,10000]
x_ticks_third_octave_labels = ["100", "200", "500", "1k", "2k", "5k", "10k"]
y_ticks_freq_db = [0,5,10,40,70,100]

third_octave_center_frequencies = [100, 125, 160, 200, 250,
            315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000, 6300, 8000] #, 10000]#, 12500] #, 16000, 20000]

third_octave_lower = [89.1, 112, 141, 178, 224, 282, 355, 447, 562, 708, 891, 1122, 1413, 1778, 2239, 2818, 3548, 4467, 5623, 7079, 8913, 11220]#, 14130, 17780,22390]
######################################

###### IR settings ###################

length_time = 0.1
stop_time = 0.021
start_time = 0.012

#start_time_FFT = 
#stop_time_FFT =


fs = 48000

x_ticks_IR = np.linspace(start_time,stop_time,5)
x_ticks_IR_labels = [str(round(i*1000,5)) for i in x_ticks_IR]

start = int(fs * start_time)
stop = int(fs * stop_time)

start_freq = 70
stop_freq = 10000




path_pos_0_75_m_box = "C:/Users/erlen/TFE4940-Master-thesis/Measurements/Scattering measurements/0 - 75 m box"
path_neg_0_75_m_box = "C:/Users/erlen/TFE4940-Master-thesis/Measurements/Scattering measurements/0 - -75 m box"
path_pos_0_75_u_box = "C:/Users/erlen/TFE4940-Master-thesis/Measurements/Scattering measurements/0 - 75 u box"
path_neg_0_75_u_box = "C:/Users/erlen/TFE4940-Master-thesis/Measurements/Scattering measurements/0 - -75 u box"
path_big_rad = "C:/Users/erlen/TFE4940-Master-thesis/Measurements/Scattering measurements/big radius neg75 - pos75"
path_big_rad_quarter = "C:/Users/erlen/TFE4940-Master-thesis/Measurements/Scattering measurements/Meas 0 - 75 0 degree LSP"

path_Simu = "C:/Users/erlen/TFE4940-Master-thesis/.venv/Simu_Meas/Simu_Meas"




def bandpass_filter(start_freq, stop_freq, sampling_rate, input_array):
    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * sampling_rate
    
    # Normalize the frequencies
    start_normalized = start_freq / nyquist_freq
    stop_normalized = stop_freq / nyquist_freq
    
    # Define the filter parameters
    filter_order = 4  # Adjust as needed
    b, a = signal.butter(filter_order, [start_normalized, stop_normalized], btype='band')
    
    # Apply the bandpass filter
    output_array = signal.lfilter(b, a, input_array)
    
    return output_array

def _nextpow2(i):
    n = 1
    while n < i : n*=2
    return n

def _Lp_from_third_oct(arr):
    """Return the sound pressure level from the third octave band array
    Args:
        arr (_type_): Third octave band array
    Returns:
        _type_: Lp
    """
    Lp = 0
    for i in arr: Lp += i 
    return Lp

def _fft_signal(array):
    #N = 48000 #
    N = _nextpow2(len(array))
    array = np.pad(array, (0,_nextpow2(len(array))-len(array)),"constant")
    y = np.fft.fft(array, N)[0:int(N/2)]/N
    #y = _runningMeanFast(y,2)
    p_y = 20*np.log10(np.abs(y))
    f = 48000*np.arange((N/2))/N
    return f, p_y

def _getFFT(arr):
    sp = np.pad(arr, (0,_nextpow2(len(arr))-len(arr)),"constant")
    sp = np.fft.fft(sp, _nextpow2(len(arr)))[0:int(_nextpow2(len(arr))/2)]# / _nextpow2(len(arr))
    sp = np.trim_zeros(sp, trim="fb")
    freq = np.fft.fftfreq(n=len(sp), d=1/fs)
    #return freq, 20*np.log10(np.abs(sp) / (2*10**(-5)))
    return np.fft.fftshift(freq), np.abs(np.fft.fftshift(sp))




def _third_Octave_bands(freq, arr, third_oct):
    third_octave_banks = []
    single_bank = []
    i = 0
    for it, x in enumerate(freq):
        
        if (x > third_oct[-1]) : break
        if (x >= third_oct[i] and x < third_oct[i+1]): 
            single_bank.append(arr[it])
        if (x >= third_oct[i+1]):
            third_octave_banks.append(single_bank)
            i += 1
            single_bank = []
            single_bank.append(arr[it])
    filtered_array = []
    for n in third_octave_banks : filtered_array.append(_Lp_from_third_oct(n))

    return filtered_array











def _SNR_func(h1, h2):
    
    return 10*np.log10(np.sum(np.power(h1 - h2,2)) / np.sum(np.power(h2,2)))


def _create_plot_IR_FFT_master_Simu():
    third_oct_arr = []
    fs_Simu = 32000
    
    #x_ticks_IR = np.linspace(0,,5)
    #x_ticks_IR_labels = [str(round(i*1000,5)) for i in x_ticks_IR]
    
    start_s = 0
    stop_s = int(4*fs_Simu)
    
    

    for i in range(1,32):
        
        # IR for all degrees
        print(i)
        IR_m_box = np.loadtxt("{0}/1_{1}_S01_R01.txt".format(path_Simu,i), dtype=float, delimiter="\t")[start_s:stop_s,:]
        IR_u_box = np.loadtxt("{0}/2_{1}_S01_R01.txt".format(path_Simu,i), dtype=float, delimiter="\t")[start_s:stop_s,:]
        
        
        
        
        
        h1 = np.abs(IR_m_box[start:stop])
        h2 = np.abs(IR_u_box[start:stop])
        
        
        # FFT for all degrees
        freq_m_box, FFT_m_box = _getFFT(np.sqrt(np.power(bandpass_filter(70, int(fs_Simu/5), fs_Simu, IR_m_box[:,1]),2)))
        freq_u_box, FFT_u_box = _getFFT(np.sqrt(np.power(bandpass_filter(70, int(fs_Simu/5), fs_Simu, IR_u_box[:,1]),2)))
        
        freq_filtered, FFT_filtered = _getFFT(np.sqrt(np.power(bandpass_filter(start_freq, int(fs_Simu/5), fs_Simu, IR_m_box[:,1] - IR_u_box[:,1]),2)))
        
        third_oct_m_box = _third_Octave_bands(freq_m_box, FFT_m_box, third_octave_lower)
        third_oct_u_box = _third_Octave_bands(freq_u_box, FFT_u_box, third_octave_lower)
        third_oct_filtered = _third_Octave_bands(freq_filtered, FFT_filtered, third_octave_lower)
        
        #fig = plt.figure(figsize=(12,7))
        fig, ax = plt.subplots(figsize=(7, 6))
        
        
        plt.style.use('ggplot')
        #ax = fig.add_subplot(121)
        #ax1 = fig.add_subplot(122)
        
        # Plot IR for the different angles
        ax.plot(IR_m_box[:,0], IR_m_box[:,1], label="IR_m_box")
        ax.plot(IR_u_box[:,0], IR_u_box[:,1], label="IR_u_box", linestyle="--" )
        #ax.plot(IR_u_box[start:,0], IR_m_box[start:,1]-IR_u_box[start:,1], label="h1 - h2")
        
        print("deg: {0} \tSNR: {1}".format(5*(i-15), round(_SNR_func(h1,h2),1)))
    
        ax.set_title("Impulse response")
        
        #ax.set_xticks(x_ticks_IR)
        #ax.set_xticklabels(x_ticks_IR_labels)
        ax.set_ylabel("Magnitude [Pa]")
        ax.set_xlabel("Time [ms]")
        ax.grid()
        ax.legend()
        
      
        # Plot FFT
        """ 
        ax1.semilogx(freq_m_box, FFT_m_box, label="FFT m box")
        ax1.semilogx(freq_u_box, FFT_u_box, label="FFT u box")
        ax1.semilogx(freq_filtered, FFT_filtered, label="FFT filtered")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_m_box, label="FFT_m_box")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_u_box, linestyle="--", label="FFT_u_box")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_filtered, label="FFT_filtered")
        ax1.set_title("Degrees of rotation: {0}".format(5*i))
        ax1.set_title("Frequency response")
        ax1.set_xlim(100,fs_Simu/5)
        ax1.set_xscale("log")
        ax1.set_xticks(x_ticks_third_octave)
        ax1.set_xticklabels(x_ticks_third_octave_labels)
        ax1.grid(which="major", color="dimgray")
        ax1.grid(which="minor", linestyle=":", color="dimgray")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude [dB]")
        """
        ax.legend()
        plt.tight_layout()
        fig.suptitle("Degrees of rotation: {0}".format(5*i))
        
        # Save each figure
        #fig.savefig("Pictures/Angle_Plot_with_box_Vs_without___{0}_Deg.png".format(i*5))
        plt.close(fig)
        #plt.show()
        third_oct_arr.append(third_oct_filtered)
    #plot_polar_scattering(np.stack(third_oct_arr), third_octave_center_frequencies[:-1])
    
    return np.stack(third_oct_arr), third_octave_center_frequencies[:-1]
    
    







def _create_plot_IR_FFT_master_Quarter():
    third_oct_arr = []
    plt.style.use("ggplot")
    for i in range(0,32):
        if i < 16:
            IR_m_box = np.loadtxt("{0}/1_{1}_S01_R01.etx".format(path_big_rad_quarter,16-i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
            IR_u_box = np.loadtxt("{0}/2_{1}_S01_R01.etx".format(path_big_rad_quarter,16-i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        
        elif i == 16:
            continue
        else:
            IR_m_box = np.loadtxt("{0}/1_{1}_S01_R01.etx".format(path_big_rad_quarter,i-15), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
            IR_u_box = np.loadtxt("{0}/2_{1}_S01_R01.etx".format(path_big_rad_quarter,i-15), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
         
         
        h1 = np.abs(IR_m_box[start:stop])
        h2 = np.abs(IR_m_box[start:stop])
            
        # FFT for all degrees
        
        freq_m_box, FFT_m_box = _getFFT(np.sqrt(np.power(bandpass_filter(100, 10000, fs, IR_m_box[start:, 1]),2)))
        freq_u_box, FFT_u_box = _getFFT(np.sqrt(np.power(bandpass_filter(100, 10000, fs, IR_u_box[start:, 1]),2))) 
        
        freq_filtered, FFT_filtered = _getFFT(np.sqrt(np.power(bandpass_filter(100, 10000, fs, IR_m_box[start:, 1] - IR_u_box[start:,1]),2)))
        
        third_oct_filtered = _third_Octave_bands(freq_filtered, FFT_filtered, third_octave_lower)
        
        #fig = plt.figure(figsize=(12,7))
        fig, ax = plt.subplots(figsize=(7, 6))
        
        
        
        #ax = fig.add_subplot(121)
        #ax1 = fig.add_subplot(122)
        
        # Plot IR for the different angles
        
        ax.plot(IR_m_box[:,0], IR_m_box[:,1], label="IR_m_box")
        ax.plot(IR_u_box[:,0], IR_u_box[:,1], label="IR_u_box")
        
        print("deg: {0} \tSNR: {1}".format(5*i, round(_SNR_func(h1,h2),1)))
        
        ax.set_title("Impulse response")
        
        ax.set_xticks(x_ticks_IR)
        ax.set_xticklabels(x_ticks_IR_labels)
        ax.set_xlim(start_time, stop_time)
        
        ax.set_ylabel("Magnitude [Pa]")
        ax.set_xlabel("Time [s]")
        #ax.grid()
        ax.legend()
        
        # Plot FFT
        """ 
        ax1.semilogx(freq_m_box, FFT_m_box, label="FFT m box")
        ax1.semilogx(freq_u_box, FFT_u_box, label="FFT u box")
        ax1.semilogx(freq_filtered, FFT_filtered-0.1, label="FFT filtered")
        
        ax1.set_title("Degrees of rotation: {0}".format(5*i))
        ax1.set_title("Frequency response")
        ax1.set_xlim(100,10000)
        ax1.set_xscale("log")
        ax1.set_xticks(x_ticks_third_octave)
        ax1.set_xticklabels(x_ticks_third_octave_labels)
        ax1.grid(which="major", color="dimgray")
        ax1.grid(which="minor", linestyle=":", color="dimgray")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude [dB]")
        ax1.legend()
        """
        plt.tight_layout()
        #fig.suptitle("Degrees of rotation: {0}".format(5*i))
        
        # Save each figure
        #fig.savefig("Quarter_meas/Angle_Plot_with_box_Vs_without___{0}_Deg.png".format(i*5))
        plt.close(fig)
        #plt.show()
        third_oct_arr.append(third_oct_filtered)
    #plot_polar_scattering(np.stack(third_oct_arr), third_octave_center_frequencies[:-1])
    
    return np.stack(third_oct_arr), third_octave_center_frequencies[:-1]

    
    




def _create_plot_IR_FFT_master_Longrange():
    third_oct_arr = []
    for i in range(0,31):
        
        # IR for all degrees
       
        IR_m_box = np.loadtxt("{0}/1_{1}_S01_R01.etx".format(path_big_rad,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_u_box = np.loadtxt("{0}/2_{1}_S01_R01.etx".format(path_big_rad,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        
        h1 = np.abs(IR_m_box[start:stop])
        h2 = np.abs(IR_u_box[start:stop])
        
        
        # FFT for all degrees
        freq_m_box, FFT_m_box = _getFFT(np.sqrt(np.power(bandpass_filter(100, 10000, fs, IR_m_box[start:,1]),2)))
        freq_u_box, FFT_u_box = _getFFT(np.sqrt(np.power(bandpass_filter(100, 10000, fs, IR_u_box[start:,1]),2)))
        
        freq_filtered, FFT_filtered = _getFFT(np.sqrt(np.power(bandpass_filter(start_freq, stop_freq, fs, IR_m_box[start:,1] - IR_u_box[start:,1]),2)))
        
        third_oct_m_box = _third_Octave_bands(freq_m_box, FFT_m_box, third_octave_lower)
        third_oct_u_box = _third_Octave_bands(freq_u_box, FFT_u_box, third_octave_lower)
        third_oct_filtered = _third_Octave_bands(freq_filtered, FFT_filtered, third_octave_lower)
        
        fig = plt.figure(figsize=(12,7))
        plt.style.use('ggplot')
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        
        # Plot IR for the different angles
        ax.plot(IR_m_box[:,0], IR_m_box[:,1], label="IR_m_box")
        ax.plot(IR_u_box[:,0], IR_u_box[:,1], label="IR_u_box", linestyle="--" )
        #ax.plot(IR_u_box[start:,0], IR_m_box[start:,1]-IR_u_box[start:,1], label="h1 - h2")
        
        print("deg: {0} \tSNR: {1}".format(5*(i-15), round(_SNR_func(h1,h2),1)))
    
        ax.set_title("Impulse response")
        
        ax.set_xticks(x_ticks_IR)
        ax.set_xticklabels(x_ticks_IR_labels)
        ax.set_ylabel("Magnitude [Pa]")
        ax.set_xlabel("Time [ms]")
        ax.grid()
        ax.legend()
        
      
        # Plot FFT
        ax1.semilogx(freq_m_box, FFT_m_box, label="FFT m box")
        ax1.semilogx(freq_u_box, FFT_u_box, label="FFT u box")
        ax1.semilogx(freq_filtered, FFT_filtered, label="FFT filtered")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_m_box, label="FFT_m_box")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_u_box, linestyle="--", label="FFT_u_box")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_filtered, label="FFT_filtered")
        ax1.set_title("Degrees of rotation: {0}".format(5*i))
        ax1.set_title("Frequency response")
        ax1.set_xlim(100,10000)
        ax1.set_xscale("log")
        ax1.set_xticks(x_ticks_third_octave)
        ax1.set_xticklabels(x_ticks_third_octave_labels)
        ax1.grid(which="major", color="dimgray")
        ax1.grid(which="minor", linestyle=":", color="dimgray")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude [dB]")
        ax1.legend()
        plt.tight_layout()
        fig.suptitle("Degrees of rotation: {0}".format(5*i))
        
        # Save each figure
        #fig.savefig("Pictures/Angle_Plot_with_box_Vs_without___{0}_Deg.png".format(i*5))
        plt.close(fig)
        #plt.show()
        third_oct_arr.append(third_oct_filtered)
    #plot_polar_scattering(np.stack(third_oct_arr), third_octave_center_frequencies[:-1])
    
    return np.stack(third_oct_arr), third_octave_center_frequencies[:-1]
        
        
        
    
    
def _normalize_and_Lp(arr):
    
    return 10*np.log10(np.power(arr/np.max(arr),2))
    



def _create_plot_IR_FFT_master():
    third_oct_arr = []
    
    for x in range(0,32):
        if x == 16:
            continue
        if x < 16:
            i = 15-x
            IR_m_box = np.loadtxt("{0}/{1}_S01_R01.etx".format(path_neg_0_75_m_box,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
            IR_u_box = np.loadtxt("{0}/{1}_S01_R01.etx".format(path_neg_0_75_u_box,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t") 
        else:
            i = x - 16
            IR_m_box = np.loadtxt("{0}/{1}_S01_R01.etx".format(path_pos_0_75_m_box,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
            IR_u_box = np.loadtxt("{0}/{1}_S01_R01.etx".format(path_pos_0_75_u_box,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        
        # Create the IR files
        h1 = np.abs(IR_m_box[start:stop])
        h2 = np.abs(IR_u_box[start:stop])
        
        
        # FFT for all degrees
        freq_m_box, FFT_m_box = _getFFT(np.sqrt(np.power(bandpass_filter(100, 10000, fs, IR_m_box[start:,1]),2)))
        freq_u_box, FFT_u_box = _getFFT(np.sqrt(np.power(bandpass_filter(100, 10000, fs, IR_u_box[start:,1]),2)))
        
        freq_filtered, FFT_filtered = _getFFT(np.sqrt(np.power(bandpass_filter(start_freq, stop_freq, fs, IR_m_box[start:,1] - IR_u_box[start:,1]),2)))
        
        third_oct_m_box = _third_Octave_bands(freq_m_box, FFT_m_box, third_octave_lower)
        third_oct_u_box = _third_Octave_bands(freq_u_box, FFT_u_box, third_octave_lower)
        third_oct_filtered = _third_Octave_bands(freq_filtered, FFT_filtered, third_octave_lower)
        
        fig = plt.figure(figsize=(12,7))
        plt.style.use('ggplot')
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        
        # Plot IR for the different angles
        ax.plot(IR_m_box[:,0], IR_m_box[:,1], label="IR_m_box")
        ax.plot(IR_u_box[:,0], IR_u_box[:,1], label="IR_u_box", linestyle="--" )
        #ax.plot(IR_u_box[start:,0], IR_m_box[start:,1]-IR_u_box[start:,1], label="h1 - h2")
        
        print("deg: {0} \tSNR: {1}".format(5*i, round(_SNR_func(h1,h2),1)))
    
        ax.set_title("Impulse response")
        
        ax.set_xticks(x_ticks_IR)
        ax.set_xticklabels(x_ticks_IR_labels)
        ax.set_ylabel("Magnitude [Pa]")
        ax.set_xlabel("Time [ms]")
        ax.grid()
        ax.legend()
        
        # Plot FFT
        ax1.semilogx(freq_m_box, FFT_m_box,label="FFT m box")
        ax1.semilogx(freq_u_box, FFT_u_box,label="FFT u box")
        ax1.semilogx(freq_filtered, FFT_filtered, label="FFT filtered")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_m_box, label="FFT_m_box")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_u_box, linestyle="--", label="FFT_u_box")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_filtered, label="FFT_filtered")
        ax1.set_title("Degrees of rotation: {0}".format(5*i))
        ax1.set_title("Frequency response")
        ax1.set_xlim(100,10000)
        ax1.set_xscale("log")
        ax1.set_xticks(x_ticks_third_octave)
        ax1.set_xticklabels(x_ticks_third_octave_labels)
        ax1.grid(which="major", color="dimgray")
        ax1.grid(which="minor", linestyle=":", color="dimgray")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude [dB]")
        ax1.legend()
        plt.tight_layout()
        fig.suptitle("Degrees of rotation: {0}".format(5*i))
        
        # Save each figure
        #fig.savefig("Pictures/Angle_Plot_with_box_Vs_without___{0}_Deg.png".format(i*5))
        plt.close(fig)
        #plt.show()
        """
        IR_pos_0_75_m_box = np.loadtxt("{0}/{1}_S01_R01.etx".format(path_pos_0_75_m_box,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_neg_0_75_m_box = np.loadtxt("{0}/{1}_S01_R01.etx".format(path_neg_0_75_m_box,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_pos_0_75_u_box = np.loadtxt("{0}/{1}_S01_R01.etx".format(path_pos_0_75_u_box,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_neg_0_75_u_box = np.loadtxt("{0}/{1}_S01_R01.etx".format(path_neg_0_75_u_box,i), dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        
        h1 = IR_pos_0_75_m_box[start:stop,1]
        h2 = IR_pos_0_75_u_box[start:stop,1]
        print("deg: {0} \tSNR: {1}".format(5*i, round(_SNR_func(h1,h2),1)))
        
        
        
        
        # FFT for the given IRs
        freq_pos_0_75_m_box, FFT_pos_m_box = _getFFT(bandpass_filter(100, 10000, fs, IR_pos_0_75_m_box[start:,1]))
        freq_neg_0_75_m_box, FFT_neg_m_box = _getFFT(bandpass_filter(100, 10000, fs, IR_neg_0_75_m_box[start:,1]))
        freq_pos_0_75_u_box, FFT_pos_u_box = _getFFT(bandpass_filter(100, 10000, fs, IR_pos_0_75_u_box[start:,1]))
        freq_neg_0_75_u_box, FFT_neg_u_box = _getFFT(bandpass_filter(100, 10000, fs, IR_neg_0_75_u_box[start:,1]))

        # FFT for the h1 - h2 as followed in ISO:17497-2
        
        freq_pos_filtered = _getFFT(bandpass_filter(start_freq, stop_freq, fs, IR_pos_0_75_m_box[start:stop,1] - IR_pos_0_75_u_box[start:stop,1]))
        freq_neg_filtered = _getFFT(bandpass_filter(start_freq, stop_freq, fs, IR_neg_0_75_m_box[start:stop,1] - IR_neg_0_75_u_box[start:stop,1]))

        # Filter into third octave bands
        third_oct_pos_m_box = _third_Octave_bands(freq_pos_0_75_m_box, FFT_pos_m_box, third_octave_lower)
        third_oct_neg_m_box = _third_Octave_bands(freq_neg_0_75_m_box, FFT_neg_m_box, third_octave_lower)
        third_oct_pos_u_box = _third_Octave_bands(freq_pos_0_75_u_box, FFT_pos_u_box, third_octave_lower)
        third_oct_neg_u_box = _third_Octave_bands(freq_neg_0_75_u_box, FFT_neg_u_box, third_octave_lower)
        
        # Initialize plot
        fig = plt.figure(figsize=(12,7))
        plt.style.use('ggplot')
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        
        # Plot IR for the different angles
        ax.plot(IR_pos_0_75_m_box[:,0],IR_pos_0_75_m_box[:,1], label="IR_pos_0_75_m_box")
        ax.plot(IR_neg_0_75_m_box[:,0],IR_neg_0_75_m_box[:,1], label="IR_neg_0_75_m_box")
        ax.plot(IR_pos_0_75_u_box[:,0],IR_pos_0_75_u_box[:,1], label="IR_pos_0_75_u_box", linestyle="--" )
        ax.plot(IR_neg_0_75_u_box[:,0],IR_neg_0_75_u_box[:,1], label="IR_neg_0_75_u_box", linestyle="--")
        
        #ax.set_title("Degrees of rotation: {0}".format(5*i))
        ax.set_title("Impulse response")
        #ax.set_xlim(0.008,0.01)
        ax.set_xticks(x_ticks_IR)
        ax.set_xticklabels(x_ticks_IR_labels)
        ax.set_ylabel("Magnitude [Pa]")
        ax.set_xlabel("Time [ms]")
        ax.grid()
        ax.legend()
        
        
        # Plot FFT
        ax1.semilogx(freq_neg_0_75_m_box, FFT_neg_m_box, label="FFT neg m box")
        ax1.semilogx(freq_pos_0_75_m_box, FFT_pos_m_box, label="FFT pos m box")
        ax1.semilogx(freq_neg_0_75_u_box, FFT_neg_u_box, label="FFT neg u box")
        ax1.semilogx(freq_pos_0_75_u_box, FFT_pos_u_box, label="FFT pos u box")
        
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_pos_m_box, label="FFT_pos_0_75_m_box")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_neg_m_box, label="FFT_neg_0_75_m_box")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_pos_u_box, linestyle="--", label="FFT_pos_0_75_u_box")
        #ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_neg_u_box, linestyle="--", label="FFT_neg_0_75_u_box")
        ax1.set_title("Degrees of rotation: {0}".format(5*i))
        ax1.set_title("Frequency response")
        ax1.set_xlim(100,10000)
        ax1.set_xscale("log")
        ax1.set_xticks(x_ticks_third_octave)
        ax1.set_xticklabels(x_ticks_third_octave_labels)
        ax1.grid(which="major", color="dimgray")
        ax1.grid(which="minor", linestyle=":", color="dimgray")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude [dB]")
        ax1.legend()
        plt.tight_layout()
        
        
        # Save each figure
        #fig.savefig("Pictures/Angle_Plot_with_box_Vs_without___{0}_Deg.png".format(i*5))
        plt.close(fig)
        plt.show()
        """
        third_oct_arr.append(third_oct_filtered)
    #plot_polar_scattering(np.stack(third_oct_arr), third_octave_center_frequencies[:-1])
    
        
        
    return np.stack(third_oct_arr), third_octave_center_frequencies[:-1]




def _create_plot_IR_FFT():
    diff_lst_ply = []
    diff_lst_exp = []
    diff_lst_pos = []
    diff_lst_neg = []
    degrees = []
    for i in range(1,16):
        file_pos_0_75_m_box = "{0}/{1}_S01_R01.etx".format(path_pos_0_75_m_box,i)
        file_neg_0_75_m_box = "{0}/{1}_S01_R01.etx".format(path_neg_0_75_m_box,i)
        file_pos_0_75_u_box = "{0}/{1}_S01_R01.etx".format(path_pos_0_75_u_box,i)
        file_neg_0_75_u_box = "{0}/{1}_S01_R01.etx".format(path_neg_0_75_u_box,i)
        
        #file_reference = "C:\Users\erlen\TFE4940-Master-thesis\Measurements\Scattering measurements\Mic response\Mic repsonse2_S01_R01.etx"
        
        #Freq_exp = np.loadtxt(file_reference, dtype=float, skiprows=24, max_rows=59500, delimiter="\t")
        
        
        IR_pos_0_75_m_box = np.loadtxt(file_pos_0_75_m_box, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_neg_0_75_m_box = np.loadtxt(file_neg_0_75_m_box, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_pos_0_75_u_box = np.loadtxt(file_pos_0_75_u_box, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_neg_0_75_u_box = np.loadtxt(file_neg_0_75_u_box, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        
        """
        freq_val = 20*np.log10(np.abs(Freq_exp[:,1]+1j*Freq_exp[:,2]))
        freq = Freq_exp[:,0]
        
        fig1 = plt.figure(figsize=(10,7))
        plt.style.use('ggplot')
        ax2 = fig1.add_subplot(111)
        ax2.semilogx(freq, freq_val, label="Freq - {0} deg".format(5*i))
        ax2.set_title("Frequency response")
        ax2.set_xlim(100,4000)
        ax2.set_ylim(-20,25)
        ax2.set_xscale("log")
        ax2.set_xticks(x_ticks_third_octave)
        ax2.set_xticklabels(x_ticks_third_octave_labels)
        ax2.grid(which="major", color="dimgray")
        ax2.grid(which="minor", linestyle=":", color="dimgray")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Magnitude [dB]")
        #ax2.legend(bbox_to_anchor=(1, 0.5), loc="center left")
        plt.close()
        #plt.show()
        _summary_
        """
        
        
        
        freq_pos_0_75_m_box, FFT_pos_m_box = _getFFT(bandpass_filter(150, 10000, fs, IR_pos_0_75_m_box[start:stop,1]))
        freq_neg_0_75_m_box, FFT_neg_m_box = _getFFT(bandpass_filter(150, 10000, fs, IR_neg_0_75_m_box[start:stop,1]))
        freq_pos_0_75_u_box, FFT_pos_u_box = _getFFT(bandpass_filter(150, 10000, fs, IR_pos_0_75_u_box[start:stop,1]))
        freq_neg_0_75_u_box, FFT_neg_u_box = _getFFT(bandpass_filter(150, 10000, fs, IR_neg_0_75_u_box[start:stop,1]))


        freq_pos_filtered = _getFFT(np.abs(bandpass_filter(start_freq, stop_freq, fs, IR_pos_0_75_m_box[start:stop,1] - IR_pos_0_75_u_box[start:stop,1])))
        freq_neg_filtered = _getFFT(np.abs(bandpass_filter(start_freq, stop_freq, fs, IR_neg_0_75_m_box[start:stop,1] - IR_neg_0_75_u_box[start:stop,1])))

        third_oct_pos_m_box = _third_Octave_bands(freq_pos_0_75_m_box, FFT_pos_m_box, third_octave_lower)
        third_oct_neg_m_box = _third_Octave_bands(freq_neg_0_75_m_box, FFT_neg_m_box, third_octave_lower)
        third_oct_pos_u_box = _third_Octave_bands(freq_pos_0_75_u_box, FFT_pos_u_box, third_octave_lower)
        third_oct_neg_u_box = _third_Octave_bands(freq_neg_0_75_u_box, FFT_neg_u_box, third_octave_lower)
        

        fig = plt.figure(figsize=(10,7))
        plt.style.use('ggplot')
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

        ax.plot(IR_pos_0_75_m_box[:,0],IR_pos_0_75_m_box[:,1], label="IR_pos_0_75_m_box")
        ax.plot(IR_neg_0_75_m_box[:,0],IR_neg_0_75_m_box[:,1], label="IR_neg_0_75_m_box")
        ax.plot(IR_pos_0_75_u_box[:,0],IR_pos_0_75_u_box[:,1], label="IR_pos_0_75_u_box", linestyle="--" )
        ax.plot(IR_neg_0_75_u_box[:,0],IR_neg_0_75_u_box[:,1], label="IR_neg_0_75_u_box", linestyle="--")
        
        #ax.set_title("Degrees of rotation: {0}".format(5*i))
        ax.set_title("Impulse response")
        ax.set_xlim(0.008,0.01)
        ax.set_xticks(x_ticks_IR)
        ax.set_xticklabels(x_ticks_IR_labels)
        ax.set_ylabel("Magnitude [Pa]")
        ax.set_xlabel("Time [ms]")
        #ax.grid()
        ax.legend()

        ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_pos_m_box, label="FFT_pos_0_75_m_box")
        ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_neg_m_box, linestyle="--", label="FFT_pos_0_75_u_box")
        ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_pos_u_box, label="FFT_neg_0_75_m_box")
        ax1.semilogx(third_octave_center_frequencies[:-1], third_oct_neg_u_box, linestyle="--",label="FFT_neg_0_75_u_box")
        #ax1.semilogx(freq_neg_0_75_m_box, (np.abs(FFT_pos_m_box)-np.abs(FFT_pos_u_box)), label="Diff. Exposed")
        #ax1.semilogx(freq_neg_0_75_m_box, (np.abs(FFT_neg_m_box)-np.abs(FFT_neg_u_box)), label="Diff. Covered")
        #ax1.set_title("Degrees of rotation for\n -Solid line: {0}\n-Dashed Line: {1}".format(5*i, 180-5*i))
        ax1.set_title("Frequency response")
        ax1.set_xlim(100,4000)
        #ax1.set_ylim(-15,30)
        ax1.set_xscale("log")
        #ax1.set_yscale("log")
        #ax1.set_ylim(0,123)
        ax1.set_xticks(x_ticks_third_octave)
        ax1.set_xticklabels(x_ticks_third_octave_labels)
        #ax1.set_yticks(y_ticks_freq_db)
        ax1.grid(which="major", color="dimgray")
        ax1.grid(which="minor", linestyle=":", color="dimgray")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude [dB]")
        ax1.legend()#bbox_to_anchor=(1, 0.5), loc="center left")
        plt.tight_layout()
        
        start_delta = int(len(freq_pos_0_75_m_box)/2) + int(300/ (freq_pos_0_75_m_box[1]-freq_pos_0_75_m_box[0]))
        stop_delta = int(len(freq_pos_0_75_m_box)/2) + int(5000 / (freq_pos_0_75_m_box[1]-freq_pos_0_75_m_box[0]))
        
        
        
        #Delta_ply = np.round(np.mean(np.abs((np.abs(FFT_neg_m_box[start_delta:stop_delta])-np.abs(FFT_neg_u_box[start_delta:stop_delta]))/np.abs(FFT_neg_m_box[start_delta:stop_delta]))),3)*100
        #Delta_abs = np.round(np.mean(np.abs(np.abs(FFT_pos_m_box[start_delta:stop_delta])-np.abs(FFT_pos_u_box[start_delta:stop_delta]))),3)
        
        Delta_ply = np.round(np.mean(np.abs((np.abs(FFT_neg_m_box[start_delta:stop_delta])-np.abs(FFT_neg_u_box[start_delta:stop_delta]))))/np.mean(np.abs(FFT_neg_m_box[start_delta:stop_delta]))*100,3)
        Delta_abs = np.round(np.mean(np.abs((np.abs(FFT_pos_m_box[start_delta:stop_delta])-np.abs(FFT_pos_u_box[start_delta:stop_delta]))))/np.mean(np.abs(FFT_neg_m_box[start_delta:stop_delta]))*100,3)
        
        Delta_meas_neg = np.round(np.mean(np.abs((np.abs(FFT_neg_u_box[start_delta:stop_delta])-np.abs(FFT_pos_u_box[start_delta:stop_delta]))))/np.mean(np.abs(FFT_neg_u_box[start_delta:stop_delta]))*100,3)
        Delta_meas_pos = np.round(np.mean(np.abs((np.abs(FFT_neg_m_box[start_delta:stop_delta])-np.abs(FFT_pos_m_box[start_delta:stop_delta]))))/np.mean(np.abs(FFT_neg_m_box[start_delta:stop_delta]))*100,3)
        
       
        
        diff_lst_ply.append(Delta_ply)
        diff_lst_exp.append(Delta_abs)
        diff_lst_pos.append(Delta_meas_pos)
        diff_lst_neg.append(Delta_meas_neg)
        
        
        degrees.append(i*5)
        #print("Start val: {0} \t stop val:{1}".format(freq_pos_0_75_m_box[start_delta],freq_pos_0_75_m_box[stop_delta]))
        #print("Start val: {0} \t stop val:{1}".format(freq_pos_0_75_m_box[start_delta1],freq_pos_0_75_m_box[stop_delta1]))
        print("Delta_ply: {0:.0%}\t Delta_abs: {1:.0%}\t Deg: {2}".format(Delta_ply,Delta_abs,i*5))
        
        
        
        fig.savefig("Pictures/Angle_Plot_with_box_Vs_without___{0}_Deg.png".format(i*5))
        plt.close(fig)
        #plt.show()
        
        #print(Exposed_IR)
    
    degrees = list(reversed(degrees))
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(degrees,diff_lst_ply, label="Diff. Covered")
    ax.plot(degrees,diff_lst_exp, label="Diff. Exposed")
    
    ax.legend()
    ax.set_title("Difference in percentage between\n85-5 degrees and 95-175 degrees")
    ax.set_xlabel("Rotation from ref. @ 90 degrees [deg]")
    ax.set_ylabel("Difference [%]")
    fig.savefig("Pictures/Difference_Normal_VS_Reversed.png")
    plt.close()
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(degrees,diff_lst_pos, label="Diff. positive")
    ax.plot(degrees,diff_lst_neg, label="Diff. negative")
    ax.legend()
    ax.set_title("Difference in percentage between\nexposed and covered frontpanel")
    ax.set_xlabel("Rotation from ref. @ 90 degrees [deg]")
    ax.set_ylabel("Difference [%]")
    fig.savefig("Difference_Covered_VS_Exposed.png")
    plt.close()



def plot_polar_scattering(scattering_data1, scattering_data2, frequencies):
    # Number of frequencies and angles
    num_frequencies = np.array(scattering_data2).shape[1]
    num_angles = np.array(scattering_data2).shape[0]
    print("num_freq = ", num_frequencies)
    print("Length_freq: ", len(scattering_data1))
    print("length frequencies: ", len(frequencies))


    # Generate polar plots for each frequency
    for i in range(num_frequencies):
        # Extract scattering data for the current frequency
        

        

        # Create figure and axes for polar plot
        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection="polar", xlim=(-90,90))

        # Plot the scattering values
        theta = np.radians(np.linspace(-75, 75, num_angles))
        
        
        scattering_values1 = _normalize_and_Lp(scattering_data1[:, i])
        scattering_values2 = _normalize_and_Lp(scattering_data2[:, i])
        #x_min = np.min(scattering_values1)*1.2 if (np.min(scattering_values1) <= np.min(scattering_values2)) else np.min(scattering_values2)*1.2
        #x_max = np.max(scattering_values1) if (np.max(scattering_values1) >= np.max(scattering_values2)) else np.max(scattering_values2)
        x_min = np.min(scattering_values1)
        x_max = np.max(scattering_values1)
        
        radi_ticks = np.linspace(x_min, x_max, 5)
        
        
        ax.plot(theta, scattering_values1, label="r=2.5 m", color='blue', alpha=0.6)
        #ax.plot(theta, scattering_values2, label="r=2.1 m simulated", color='green', alpha=0.6)
        #ax.fill(theta, scattering_values1, color='blue', alpha=0.3)
        #ax.fill(theta, scattering_values2, color='green', alpha=0.3)
        
        ax.fill_between(theta, scattering_values1, [x_min], interpolate=True, color='blue', alpha=0.3)
        #ax.fill_between(theta, scattering_values2, [x_min], interpolate=True, color='green', alpha=0.3)

        # Set plot attributes
        #ax.set_yticklabels([]) 
        plt.ylim(x_min, x_max)  # Set custom radius limits

        plt.yticks(np.round(radi_ticks,1), [str(r) for r in np.round(radi_ticks,1)]) 
        # Set title and labels
        frequency = frequencies[i]
        #ax.set_title(f'Scattering at {frequency} Hz')
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_theta_offset(.5*np.pi)
        ax.spines['polar'].set_visible(False) 
        #plt.legend()
        
        # Save the figure with the frequency value as the filename
        filename = f'filtered scattering_Second_{frequency}Hz.png'
        print("saved: ", f' filtered scattering_{frequency}Hz.png')
        plt.savefig(filename)
        #plt.show()
        # Close the figure to avoid memory leaks
        plt.close(fig)




def plot_polar_scattering_Quarter(scattering_data1, scattering_data2, frequencies):
    # Number of frequencies and angles
    num_frequencies = np.array(scattering_data2).shape[1]
    num_angles = np.array(scattering_data2).shape[0]

    # Generate polar plots for each frequency
    for i in range(num_frequencies):
        # Extract scattering data for the current frequency
        

        

        # Create figure and axes for polar plot
        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection="polar", xlim=(-90,90))

        # Plot the scattering values
        theta = np.radians(np.linspace(-75, 75, num_angles))
        
        
        scattering_values1 = _normalize_and_Lp(scattering_data1[:, i])
        scattering_values2 = _normalize_and_Lp(scattering_data2[:, i])
        x_min = np.min(scattering_values1)*1.2 if (np.min(scattering_values1) <= np.min(scattering_values2)) else np.min(scattering_values2)*1.2
        x_max = np.max(scattering_values1) if (np.max(scattering_values1) >= np.max(scattering_values2)) else np.max(scattering_values2)
        radi_ticks = np.linspace(x_min, x_max, 5)
        
        
        ax.plot(theta, scattering_values1, label="r=1.1 m", color='blue', alpha=0.6)
        ax.plot(theta, scattering_values2, label="r=2.1 m", color='green', alpha=0.6)
        #ax.fill(theta, scattering_values1, color='blue', alpha=0.3)
        #ax.fill(theta, scattering_values2, color='green', alpha=0.3)
        
        
        ax.fill_between(theta, scattering_values1, [x_min], interpolate=True, color='blue', alpha=0.3)
        ax.fill_between(theta, scattering_values2, [x_min], interpolate=True, color='green', alpha=0.3)

        # Set plot attributes
        #ax.set_yticklabels([]) 
        plt.ylim(x_min, x_max)  # Set custom radius limits

        plt.yticks(np.round(radi_ticks,1), [str(r) for r in np.round(radi_ticks,1)]) 
        # Set title and labels
        frequency = frequencies[i]
        ax.set_title(f'Scattering at {frequency} Hz')
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_theta_offset(.5*np.pi)
        ax.spines['polar'].set_visible(False) 
        plt.legend()
        
        # Save the figure with the frequency value as the filename
        filename = f'filtered scattering_{frequency}Hz.png'
        print("saved: ", f' filtered scattering_{frequency}Hz.png')
        plt.savefig(filename)

        # Close the figure to avoid memory leaks
        plt.close(fig)



def directional_diffusion_coefficient(arr1, arr2, arr3, arr_simu):
    d_theta1 = []
    d_theta2 = []
    d_theta3 = []
    d_theta_simu = []
    num_frequencies = np.array(arr1).shape[1]
    num_angles = np.array(arr1).shape[0]
    
    for i in range(num_frequencies):
        arr1_Lp = _normalize_and_Lp(arr1[:, i])
        arr2_Lp = _normalize_and_Lp(arr2[:, i])
        arr3_Lp = _normalize_and_Lp(arr3[:, i])
        arr_simu_Lp = _normalize_and_Lp(arr_simu[:, i])

        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp_simu = 0
        temp11 = 0
        temp12 = 0
        temp13 = 0
        temp_simu2 = 0
        for j in range(num_angles):
            temp1 += 10**(arr1_Lp[j]/10)
            temp2 += 10**(arr2_Lp[j]/10)
            temp3 += 10**(arr3_Lp[j]/10)
            temp_simu += 10**(arr_simu_Lp[j]/10)
            temp11 += (10**(arr1_Lp[j]/10))**2
            temp12 += (10**(arr2_Lp[j]/10))**2
            temp13 += (10**(arr3_Lp[j]/10))**2
            temp_simu2 += (10**(arr_simu_Lp[j]/10))**2
            
        d_theta1.append((temp1**2 - temp11) / ((num_angles-1)*temp11))
        d_theta2.append((temp2**2 - temp12) / ((num_angles-1)*temp12))
        d_theta3.append((temp3**2 - temp13) / ((num_angles-1)*temp13))
        d_theta_simu.append((temp_simu**2 - temp_simu2) / ((num_angles-1)*temp_simu2))
    
    fig, ax = plt.subplots(figsize=(9, 7))

    plt.style.use("bmh")
    
    
    coeff1 = np.polyfit(third_octave_center_frequencies, d_theta1, 10)
    polynomial1 = np.poly1d(coeff1)
    coeff2 = np.polyfit(third_octave_center_frequencies, d_theta2, 10)
    polynomial2 = np.poly1d(coeff2)
    coeff3 = np.polyfit(third_octave_center_frequencies, d_theta3, 10)
    polynomial3 = np.poly1d(coeff3)
    coeff4 = np.polyfit(third_octave_center_frequencies, d_theta_simu, 10)
    polynomial4 = np.poly1d(coeff4)
    temp = []
    
    
    
    y_line1=polynomial1(third_octave_center_frequencies)
    y_line2=polynomial2(third_octave_center_frequencies)
    y_line3=polynomial3(third_octave_center_frequencies)
    y_line4=polynomial4(third_octave_center_frequencies)
    
    average = (np.array(d_theta1) + np.array(d_theta2) + np.array(d_theta3)) / 3
    
    STD = np.std([np.array(d_theta1), np.array(d_theta2), np.array(d_theta3)], axis=0)
       
    #ax.semilogx(third_octave_center_frequencies, d_theta1, label="radius=1.1 m", marker="v")
    #ax.semilogx(third_octave_center_frequencies, d_theta2, label="radius=2.0 m", marker="o")
    #ax.semilogx(third_octave_center_frequencies, d_theta3, label="radius=2.5 m", marker="s")
    ax.semilogx(third_octave_center_frequencies, d_theta_simu, label="radius=2.5 m - Simulated", marker="p")
    ax.semilogx(third_octave_center_frequencies, average, label="Average diffusion coefficient", marker="v")
    ax.fill_between(third_octave_center_frequencies, average - STD, average + STD, alpha=0.3, label="STD" , color="blue")
    
    #ax.semilogx(third_octave_center_frequencies, y_line1, label="Fitted line r=1.1 m")
    #ax.semilogx(third_octave_center_frequencies, y_line2, label="Fitted line r=2.0 m")
    #ax.semilogx(third_octave_center_frequencies, y_line3, label="Fitted line r=2.5 m")
    #ax.semilogx(third_octave_center_frequencies, y_line4, label="Fitted line r=2.5 m simulated")
    
    ax.legend()
    ax.set_xticks(x_ticks_third_octave)
    ax.set_xticklabels(x_ticks_third_octave_labels)


    ax.grid(which="major", color="dimgray")
    ax.grid(which="minor", linestyle=":", color="dimgray")
    ax.set_ylabel("[$d_{\u03b8}$]")
    ax.set_xlabel("Frequency [Hz]")
    #ax.set_title("Directional Diffusion Coeficient [$d_{\u03b8}$]")
    plt.show()   




third_oct_val_simu, third_oct_freq_simu = _create_plot_IR_FFT_master_Simu()
third_oct_val_smal_radius, third_oct_freq_smal_radius = _create_plot_IR_FFT_master()

third_oct_val_big_radius, third_oct_freq_big_radius = _create_plot_IR_FFT_master_Longrange()

third_oct_val_big_radius_quarter, third_oct_freq_big_radius_quarter = _create_plot_IR_FFT_master_Quarter()



directional_diffusion_coefficient(third_oct_val_smal_radius, third_oct_val_big_radius, third_oct_val_big_radius_quarter, third_oct_val_simu)

plot_polar_scattering(third_oct_val_big_radius, third_oct_val_simu, third_octave_center_frequencies)

    


