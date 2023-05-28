import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from scipy.signal import butter, filtfilt

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


delta_d = 0.25  # in meters
tot_time = 1.0   # in seconds
energy_time = 0.05  # in seconds
freq = 50
l = 4.0
width = 3.0
height = 12.5

impulse = True
harmonic = False
sweep = False
source_position = np.array([1.2, 1.1, 1.5])

R = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

runtime = True
Temp = 291
ρ_air = 1.225
c = 343.2 * np.sqrt(Temp / 293)
delta_t = (delta_d / c)
fs = int(1 / delta_t)
p0 = 2e-5
N = int(tot_time / delta_t)
nx = int(l / delta_d + 1)
ny = int(width / delta_d + 1)
nz = int(height / delta_d + 1)
meas_pos1 = np.array([1.0, 1.0, 1.0])
meas_pos2 = np.array([1.8, 1.2, 1.2])
meas_pos3 = np.array([2.2, 1.2, 7.2])
mic1 = []
mic2 = []
mic3 = []
mic1_Lp = []
mic2_Lp = []
mic3_Lp = []




##################################################################


Labeled_tlm = np.zeros((nx, ny, nz), dtype=int)
pressure_grid = []
SN = np.zeros((nx, ny, nz))
SE = np.zeros((nx, ny, nz))
SS = np.zeros((nx, ny, nz))
SW = np.zeros((nx, ny, nz))
SU = np.zeros((nx, ny, nz))
SD = np.zeros((nx, ny, nz))
IN = np.zeros((nx, ny, nz))
IE = np.zeros((nx, ny, nz))
IS = np.zeros((nx, ny, nz))
IW = np.zeros((nx, ny, nz))
IU = np.zeros((nx, ny, nz))
ID = np.zeros((nx, ny, nz))








def propagate():
    global IN, IE, IS, IW, IU, ID
    for i in range(2,Labeled_tlm.shape[0]-2):
        for j in range(2,Labeled_tlm.shape[1]-2):
            for k in range(2,Labeled_tlm.shape[2]-2):
               
                case_used, case_unused, diffusor = case(Labeled_tlm[i, j, k])
            
                for n in case_used:
                    Refl = R[6] if diffusor else R[n]
                    
                    if case_used == 0: 
                        IN[i, j, k] = SS[i, j - 1, k]
                        IE[i, j, k] = SW[i + 1, j, k]
                        IS[i, j, k] = SN[i, j + 1, k]
                        IW[i, j, k] = SE[i - 1, j, k]
                        IU[i, j, k] = SD[i, j, k + 1]
                        ID[i, j, k] = SU[i, j, k - 1]
                        continue
                    
                    IN[i, j, k] = Refl * SN[i, j, k] if 1 in case_used else SS[i, j - 1, k]
                    
                    IS[i, j, k] = Refl * SS[i, j, k] if 2 in case_used else SN[i, j + 1, k]
                    
                    IW[i, j, k] = Refl * SW[i, j, k] if 3 in case_used else SE[i - 1, j, k]
                    
                    IE[i, j, k] = Refl * SE[i, j, k] if 4 in case_used else SW[i + 1, j, k]
                    
                    ID[i, j, k] = Refl * SD[i, j, k] if 5 in case_used else SU[i, j, k - 1]
                    
                    IU[i, j, k] = Refl * SU[i, j, k] if 6 in case_used else SD[i, j, k + 1]
    return



def case(n):
    diffusor = False
    if n < 0:
        diffusor = True
        n = abs(n)
    
    num_str = str(n)
    used_nums = [int(digit) for digit in num_str]
    all_nums = set(range(0, 7))
    unused_nums = all_nums.difference(used_nums)
    
    return used_nums, unused_nums, diffusor



def scatter():
    global SW, SN, SE, SS, SU, SD
    SW = (1/3) * ((-2 * IW) + IN + IE + IS + IU + ID)
    SN = (1/3) * (IW + (-2 * IN) + IE + IS + IU + ID)
    SE = (1/3) * (IW + IN + (-2 * IE) + IS + IU + ID)
    SS = (1/3) * (IW + IN + IE + (-2 * IS) + IU + ID)
    SU = (1/3) * (IW + IN + IE + IS + (-2 * IU) + ID)
    SD = (1/3) * (IW + IN + IE + IS + IU + (-2 * ID))
    
    return



def overal_pressure():
    global pressure_grid
    IN_temp = IN#[1:-1,1:-1,1:-1]
    IE_temp = IE#[1:-1,1:-1,1:-1]
    IS_temp = IS#[1:-1,1:-1,1:-1]
    IW_temp = IW#[1:-1,1:-1,1:-1]
    IU_temp = IU#[1:-1,1:-1,1:-1]
    ID_temp = ID#[1:-1,1:-1,1:-1]
    sum_nodes = (1/3) * (IN_temp + IE_temp + IS_temp + IW_temp + IU_temp + ID_temp)
    pressure_grid.append(sum_nodes)
    return





def insert_energy(val):
    global IN, IE, IS, IW, IU, ID
    source_pos = _real_to_index(source_position)
    val = val * 10
    IN[source_pos[0], source_pos[1] + 1, source_pos[2]] = val
    IE[source_pos[0] - 1, source_pos[1], source_pos[2]] = val
    IS[source_pos[0], source_pos[1] - 1, source_pos[2]] = val
    IW[source_pos[0] + 1, source_pos[1], source_pos[2]] = val
    IU[source_pos[0], source_pos[1], source_pos[2] - 1] = val
    ID[source_pos[0], source_pos[1], source_pos[2] + 1] = val



def update_incident_with_energy(it):
    if delta_t * it >= energy_time:
        return
    if harmonic:
        insert_energy(harmonic_val(it))
    elif impulse:
        insert_energy(gaussian_pulse(it, int(energy_time / delta_t)-5, 4))
    return

    
    
    
def harmonic_val(it):
    return 40 * np.cos(it * delta_t * 2 * np.pi * freq)   

def gaussian_pulse(timestep, peak_time, sigma):
    amplitude = 10.0  # Amplitude of the Gaussian pulse
    mean = peak_time  # Mean (center) of the Gaussian pulse

    # Calculate the value of the Gaussian pulse at the given timestep
    value = amplitude * math.exp(-((timestep - mean) ** 2) / (2 * sigma ** 2))

    return value


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




def run_simulation():
    for it in tqdm(range(0, N+1)):
        # Calculate values for the next timestep
        calculate_next_step(it)

        # Sum overall pressure for each node
        overal_pressure()
        # readline()

    meas_pressure()
    #print(plot_fft(mic3, fs, energy_time))
    plot_data(ltau(mic1, fs, delta_t * 10, p0), ltau(mic2, fs, delta_t * 10, p0), ltau(mic3, fs, delta_t * 10, p0))




def meas_pressure():
    global mic1, mic2, mic3
    for item in pressure_grid:
        mic1.append(item[_real_to_index(meas_pos1)[0], _real_to_index(meas_pos1)[1], _real_to_index(meas_pos1)[2]])
        mic2.append(item[_real_to_index(meas_pos2)[0], _real_to_index(meas_pos2)[1], _real_to_index(meas_pos2)[2]])
        mic3.append(item[_real_to_index(meas_pos3)[0], _real_to_index(meas_pos3)[1], _real_to_index(meas_pos3)[2]])




def calculate_next_step(it):
    # Insert energy into
    update_incident_with_energy(it)

    # Update scatter values
    scatter()

    # Update incident pulses
    propagate()





def tlmGrid():
    global Labeled_tlm
    print("Creating ShoeLabeled_tlm shape")
    print("d:", delta_d, "m")
    print("size:\t x_direction:", nx*delta_d, "m \t y_direction:", ny*delta_d, "m \t z_direction:", nz*delta_d, "m")

    # Set surface values by checking the position
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
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

    # Set Edge values
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
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

    # Set Corner values
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
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

    return





def _real_to_index(val):
    ind = [int(val[0] / delta_d) - 1, int(val[1] / delta_d) - 1, int(val[2] / delta_d) - 1]
    return ind

def harmonic_val(it):
    return 40 * math.cos(it * delta_t * 2 * math.pi * freq)






def plot_data(arr1, arr2, arr3):
    print("Plotting arrays")
    t = np.arange(0, N) / fs


    plt.plot(t, arr1, label="Mic 1")
    plt.plot(t, arr2, label="Mic 2")
    plt.plot(t, arr3, label="Mic 3")
    plt.style.use('ggplot')
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Time weighted sound pressure level at three positions")
    #plt.ylim(0, 125)
    plt.legend()
    plt.show()
    plt.savefig("Python_testing_boundaries top boundary only.png")





if __name__ == '__main__':
    tlmGrid()
    # display(Labeled_tlm)
    run_simulation()


