import numpy as np
import numba as nb
import matplotlib.pyplot as plt
#from numba import jit
#import numpy as np
import time
from scipy.signal import butter, filtfilt
import math


####### Font values #######
SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
###########################

Temp = 291
ρ_air = 1.225
c = 343.2 * np.sqrt(Temp / 293)

tot_time = 1.0   # in seconds
energy_time = 0.2  # in seconds
freq = 250
l = 14.0
width = 7.0
height = 10.0
delta_d = c / (freq * 10)  # in meters
impulse = False
harmonic = True
sweep = False
source_position = np.array([1.2, 1.1, 1.5])

R = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])

runtime = True

delta_t = (delta_d / c)
fs = int(1 / delta_t)
p0 = 2e-5
N = int(tot_time / delta_t)
nx = int(l / delta_d + 1)
ny = int(width / delta_d + 1)
nz = int(height / delta_d + 1)
meas_pos1 = np.array([1.0, 1.0, 1.0])
meas_pos2 = np.array([1.8, 1.2, 1.2])
meas_pos3 = np.array([2.2, 1.2, 2.2])
mic1 = np.zeros(N)
mic2 = np.zeros(N)
mic3 = np.zeros(N)
mic_overal = np.zeros(N)
mic1_Lp = []
mic2_Lp = []
mic3_Lp = []
##################################################################

 # Creation of variables used

pressure_grid = np.zeros((N, nx, ny, nz), dtype=np.float64)




@nb.jit(nopython=True)
def case(n):
    #case_nums = case_nums*0
    num_digits = int(np.log10(n)) + 1

    arr = np.zeros((num_digits), dtype=np.int64)

    for i in range(num_digits - 1, -1, -1):
        arr[i] = n % 10
        n //= 10
    return arr

@nb.jit(nopython=True)
def propagate(Labeled_tlm, SN, SE, SS, SW, SU, SD, IN, IE, IS, IW, IU, ID, R):
    
    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                if Labeled_tlm[i,j,k] == 0: 
                    case_used = np.array([0]) 
                else: 
                    case_used = case(Labeled_tlm[i, j, k])
                for n in case_used:
                    Refl = 0 if 0 in case_used else R[n]
                    IN[i, j, k] = Refl * SN[i, j, k] if 1 in case_used else SS[i, j - 1, k]
                    IS[i, j, k] = Refl * SS[i, j, k] if 2 in case_used else SN[i, j + 1, k]
                    IW[i, j, k] = Refl * SW[i, j, k] if 3 in case_used else SE[i - 1, j, k]
                    IE[i, j, k] = Refl * SE[i, j, k] if 4 in case_used else SW[i + 1, j, k]
                    ID[i, j, k] = Refl * SD[i, j, k] if 5 in case_used else SU[i, j, k - 1]
                    IU[i, j, k] = Refl * SU[i, j, k] if 6 in case_used else SD[i, j, k + 1]
    return IN, IE, IS, IW, IU, ID

@nb.jit(nopython=True)
def scatter(SN, SE, SS, SW, SU, SD, IN, IE, IS, IW, IU, ID):
    
    SW = (1/3) * ((-2 * IW) + IN + IE + IS + IU + ID)
    SN = (1/3) * (IW + (-2 * IN) + IE + IS + IU + ID)
    SE = (1/3) * (IW + IN + (-2 * IE) + IS + IU + ID)
    SS = (1/3) * (IW + IN + IE + (-2 * IS) + IU + ID)
    SU = (1/3) * (IW + IN + IE + IS + (-2 * IU) + ID)
    SD = (1/3) * (IW + IN + IE + IS + IU + (-2 * ID))
    return SN, SE, SS, SW, SU, SD

@nb.jit(nopython=True)
def overal_pressure(IN, IE, IS, IW, IU, ID):
    IN_temp = IN
    IE_temp = IE
    IS_temp = IS
    IW_temp = IW
    IU_temp = IU
    ID_temp = ID
    sum_nodes = (1/3) * (IN_temp + IE_temp + IS_temp + IW_temp + IU_temp + ID_temp)
    return sum_nodes

@nb.jit(nopython=True)
def insert_energy(val, IN, IE, IS, IW, IU, ID):
    
    source_pos = _real_to_index(source_position)
    val = val * 10
    IN[source_pos[0], source_pos[1], source_pos[2]] = val
    IE[source_pos[0], source_pos[1], source_pos[2]] = val
    IS[source_pos[0], source_pos[1], source_pos[2]] = val
    IW[source_pos[0], source_pos[1], source_pos[2]] = val
    IU[source_pos[0], source_pos[1], source_pos[2]] = val
    ID[source_pos[0], source_pos[1], source_pos[2]] = val
    return IN, IE, IS, IW, IU, ID

@nb.jit(nopython=True)
def update_incident_with_energy(it, IN, IE, IS, IW, IU, ID):
    if delta_t * it >= energy_time:
        return
    if harmonic:
        IN, IE, IS, IW, IU, ID = insert_energy(harmonic_val(it), IN, IE, IS, IW, IU, ID)
    elif impulse:
        IN, IE, IS, IW, IU, ID = insert_energy(gaussian_pulse(it, int(energy_time / delta_t) - 5, 4), IN, IE, IS, IW, IU, ID)
    return IN, IE, IS, IW, IU, ID

@nb.jit(nopython=True)
def calculate_next_step(R, Labeled_tlm, it, IN, IE, IS, IW, IU, ID, SN, SE, SS, SW, SU, SD):
    IN, IE, IS, IW, IU, ID = update_incident_with_energy(it, IN, IE, IS, IW, IU, ID)
    SN, SE, SS, SW, SU, SD = scatter(SN, SE, SS, SW, SU, SD, IN, IE, IS, IW, IU, ID)
    IN, IE, IS, IW, IU, ID = propagate(Labeled_tlm, SN, SE, SS, SW, SU, SD, IN, IE, IS, IW, IU, ID, R)
    return IN, IE, IS, IW, IU, ID, SN, SE, SS, SW, SU, SD


def run_simulation():
    SN = np.zeros((nx, ny, nz), dtype=np.float64)
    SE = np.zeros((nx, ny, nz), dtype=np.float64)
    SS = np.zeros((nx, ny, nz), dtype=np.float64)
    SW = np.zeros((nx, ny, nz), dtype=np.float64)
    SU = np.zeros((nx, ny, nz), dtype=np.float64)
    SD = np.zeros((nx, ny, nz), dtype=np.float64)
    IN = np.zeros((nx, ny, nz), dtype=np.float64)
    IE = np.zeros((nx, ny, nz), dtype=np.float64)
    IS = np.zeros((nx, ny, nz), dtype=np.float64)
    IW = np.zeros((nx, ny, nz), dtype=np.float64)
    IU = np.zeros((nx, ny, nz), dtype=np.float64)
    ID = np.zeros((nx, ny, nz), dtype=np.float64)
    Labeled_tlm = tlmGrid().astype(np.int64)
    R = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    for it in range(0, N+1):
        IN, IE, IS, IW, IU, ID, SN, SE, SS, SW, SU, SD = calculate_next_step(R, Labeled_tlm, it, IN, IE, IS, IW, IU, ID, SN, SE, SS, SW, SU, SD)
        #temp_sum = overal_pressure(IN, IE, IS, IW, IU, ID)
        #pressure_grid[it] = temp_sum
        
    return #pressure_grid
    





def meas_pressure(mic1, mic2, mic3, mic_overal, pressure_grid):
    for num, item in enumerate(pressure_grid):
        mic1[num] = item[_real_to_index(meas_pos1)[0], _real_to_index(meas_pos1)[1], _real_to_index(meas_pos1)[2]]
        mic2[num] = item[_real_to_index(meas_pos2)[0], _real_to_index(meas_pos2)[1], _real_to_index(meas_pos2)[2]]
        mic3[num] = item[_real_to_index(meas_pos3)[0], _real_to_index(meas_pos3)[1], _real_to_index(meas_pos3)[2]]
        mic_overal[num] = np.sqrt(np.sum(np.power(np.array(item), 2))) / ((nx)*(ny)*(nz))
    return mic1, mic2, mic3, mic_overal
    
    
@nb.jit(nopython=True)
def tlmGrid():
    Labeled_tlm = np.zeros((nx, ny, nz), dtype=np.int64)
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                if i == 0:
                    Labeled_tlm[i, j, k] = 3
                elif i == nx-1:
                    Labeled_tlm[i, j, k] = 4
                elif j == 0:
                    Labeled_tlm[i, j, k] = 1
                elif j == ny-1:
                    Labeled_tlm[i, j, k] = 2
                elif k == 0:
                    Labeled_tlm[i, j, k] = 5
                elif k == nz-1:
                    Labeled_tlm[i, j, k] = 6

    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                if i == 0 and j == ny-1:
                    Labeled_tlm[i, j, k] = 23
                elif j == ny-1 and k == 0:
                    Labeled_tlm[i, j, k] = 25
                elif i == nx-1 and j == ny-1:
                    Labeled_tlm[i, j, k] = 24
                elif j == ny-1 and k == nz-1:
                    Labeled_tlm[i, j, k] = 26
                elif i == nx-1 and k == 0:
                    Labeled_tlm[i, j, k] = 45
                elif i == nx-1 and k == nz-1:
                    Labeled_tlm[i, j, k] = 46
                elif i == nx-1 and j == 0:
                    Labeled_tlm[i, j, k] = 14
                elif j == 0 and k == 0:
                    Labeled_tlm[i, j, k] = 15
                elif i == 0 and k == 0:
                    Labeled_tlm[i, j, k] = 35
                elif i == 0 and k == nz-1:
                    Labeled_tlm[i, j, k] = 36
                elif i == 0 and j == 0:
                    Labeled_tlm[i, j, k] = 13
                elif j == 0 and k == nz-1:
                    Labeled_tlm[i, j, k] = 16

    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                if i == nx-1 and j == 0 and k == 0:
                    Labeled_tlm[i, j, k] = 145
                elif i == 0 and j == 0 and k == 0:
                    Labeled_tlm[i, j, k] = 135
                elif i == 0 and j == ny-1 and k == 0:
                    Labeled_tlm[i, j, k] = 125
                elif i == nx-1 and j == ny-1 and k == 0:
                    Labeled_tlm[i, j, k] = 245
                elif i == nx-1 and j == 0 and k == nz-1:
                    Labeled_tlm[i, j, k] = 146
                elif i == 0 and j == 0 and k == nz-1:
                    Labeled_tlm[i, j, k] = 136
                elif i == 0 and j == ny-1 and k == nz-1:
                    Labeled_tlm[i, j, k] = 126
                elif i == nx-1 and j == ny-1 and k == nz-1:
                    Labeled_tlm[i, j, k] = 246

    return Labeled_tlm



def ltau(p, fs, τ, pref=p0):
    bufsize = int(np.floor(0.5 * τ * fs))  # number of samples to extrapolate
    p2 = np.array(p)**2  # squaring the signals
    # fictitious samples are added at the beginning to avoid very negative sound levels right after t=0
    p2 = np.concatenate((np.mean(p2[:bufsize]) * np.ones(bufsize), p2))
    # Implementing exponential time weighting as a low pass filter
    nyquist = fs / 2
    cutoff = 1 / (2 * np.pi * τ)
    b, a = butter(1, cutoff / nyquist, 'low')
    y = filtfilt(b, a, p2)  # filtering
    y = 1 / (p0 ** 2) * y[bufsize + 1:]  # adjusting the length to that of the input vector
    l = 10 * np.log10(y)
    return l

@nb.jit(nopython=True)
def harmonic_val(it):
    return 40 * np.cos(it * delta_t * 2 * np.pi * freq)   

@nb.jit(nopython=True)
def gaussian_pulse(timestep, peak_time, sigma):
    amplitude = 10.0  # Amplitude of the Gaussian pulse
    mean = peak_time  # Mean (center) of the Gaussian pulse

    # Calculate the value of the Gaussian pulse at the given timestep
    value = amplitude * math.exp(-((timestep - mean) ** 2) / (2 * sigma ** 2))

    return value

@nb.jit(nopython=True)
def _real_to_index(val):
    ind = [int(val[0] / delta_d) - 1, int(val[1] / delta_d) - 1, int(val[2] / delta_d) - 1]
    return ind

@nb.jit(nopython=True)
def harmonic_val(it):
    return 40 * math.cos(it * delta_t * 2 * math.pi * freq)



def plot_data(arr1, arr2, arr3, arr4):
    print("Plotting arrays")
    t = np.arange(0, N) / fs


    plt.plot(t, arr1, label="Mic 1")
    plt.plot(t, arr2, label="Mic 2")
    plt.plot(t, arr3, label="Mic 3")
    plt.plot(t, 20*np.log10(arr4 / p0)[:-1], label="Overall pressure")
    plt.style.use('ggplot')
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Time weighted sound pressure level at three positions")
    #plt.ylim(0, 125)
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("Python_testing_boundaries top boundary only.png")

    
    
    
   
    
    
# The simulation itself
start = time.time()
run_simulation()


end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))


# Plotting of data
#mic1, mic2, mic3, mic_overal = meas_pressure(mic1, mic2, mic3, mic_overal, pressure_grid)
#plot_data(ltau(mic1, fs, delta_t * 10, p0), ltau(mic2, fs, delta_t * 10, p0), ltau(mic3, fs, delta_t * 10, p0), np.array(mic_overal))

