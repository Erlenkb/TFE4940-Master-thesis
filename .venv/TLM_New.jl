using FFTW, Plots
using ProgressMeter
using DSP
using Statistics, StatsPlots
using Printf
using PlotThemes

"""
When separating functions into files, include them here. 

Currently this file only need ´ltau.jl´ to work, and the packages above
"""
#include("Draw_box_closed.jl")
include("ltau.jl")
#include("TLM operations.jl")
#include("Place objects into TLM.jl")
#include("Acoustical operations.jl")




"""
Variables that can be altered by the user:

tot_time: Total simulation time in seconds
energy_time: Time for the source signal to be one
freq: If ´harmonic´ is true, this variable set the frequency of the signal
l: Length of the room in meters
width: Width of the room in meters
height: Height of the room in meters
source_position: the carteesian coordinates in meters for where the source signal should be.
R: Reflection coefficent values for the grid. first 6 elements are for the direction, related to the rule numbers (i.e 1=north, 2=south, 3=west, 4=east, 5=down, 6=up)
    the 7th element is for the diffusor.
Temp: Temperature within the grid given in Kelvin
meas_pos1,2&3: Measurement position for the three independent microphone positions
fs: Sampling frequency

´pressure_grid´ is a pressure array storing all pressure values for the entirety of the simulation. Use elements in that grid to process data at the position of interest
"""
tot_time = 1.0   # in seconds
energy_time = 0.2  # in seconds
freq = 150
l = 4
width = 3
height = 2.5

impulse = false
harmonic = true
sweep = false
source_position = [2.5,1.3,1.4]
R = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
runtime = true
Temp = 291
meas_pos1 = [1.5, 1.4, 1.4]
meas_pos2 = [1.5, 1.4, 1.4]
meas_pos3 = [1.5, 1.4, 1.4]
fs = 3500



####### Box parameters ###############
d_box = 0.25
b_box = 0.3
h_box = 0.45
N_QRD = 7
d_max_QRD = 0.1
d_vegger = 0.02
d_bakplate = 0.02
d_skilleplate = 0.02
d_absorbent_QRD = 0.012
b_QRD = (b_box-2*d_vegger) / 7
#######################################


##### Instanciate global variables
ρ_air = 1.225
c = 343.2*sqrt(Temp/293)/(1)
theme(:wong2)
Δd = c / fs
Δt = (Δd / c)
p0 = 2e-5
N = Int(tot_time ÷ Δt)
nx = Int(l ÷ Δd + 1)
ny = Int(width ÷ Δd + 1)
nz = Int(height ÷ Δd + 1)
room_modes = [128.63, 171.5, 214,  250]
mic1 = []
mic2 = []
mic3 = []
mic1_Lp = []
mic2_Lp = []
mic3_Lp = []
gif_mic = zeros((nx,ny,N))
Labeled_tlm = zeros(Int, (nx,ny,nz))
pressure_grid = [] 
SN = zeros((nx,ny,nz))
SE = zeros((nx,ny,nz))
SS = zeros((nx,ny,nz))
SW = zeros((nx,ny,nz))
SU = zeros((nx,ny,nz))
SD = zeros((nx,ny,nz))
IN = zeros((nx,ny,nz))
IE = zeros((nx,ny,nz))
IS = zeros((nx,ny,nz))
IW = zeros((nx,ny,nz))
IU = zeros((nx,ny,nz))
ID = zeros((nx,ny,nz))



"""
    improvised_gaussian()

    improvised gaussian signal created from a binary signal and tukey window. only used for fun
"""
function improvised_gaussian()
    N = length(collect(0:1/fs:energy_time))
    amplitude = 1.0
    mean = (N + 1) / 2  # The mean is in the center of the pulse
    sigma = N / 6      # Standard deviation for approximately 3 standard deviations

    # Create an array of length N and initialize it with zeros
    pulse = zeros(N)

    # Calculate the Gaussian pulse values and store them in the array
    for t in 1:N
        pulse[t] = amplitude * exp(-((t - mean) ^ 2) / (2 * sigma ^ 2))
    end
    
    return pulse
end


"""
    generateSweep(start_freq, end_freq, duration, sample_rate)

    Takes in the start and end frequency value and return a list containing a sweep filtered using a tukey window to ensure stability
"""
function generateSweep(start_freq, end_freq, duration, sample_rate)
    t = collect(0:1/sample_rate:duration)
    freq_range = end_freq - start_freq
    phase = 2π * (start_freq * t + 0.5 * freq_range * t.^2 / duration)
    sweep_signal = sin.(phase)
    
    window = tukey(length(sweep_signal), 0.05)  
    sweep_signal = sweep_signal .* window
    
    return sweep_signal
end



"""
    Reflection_coefficient(Γ)

    Return the reflection coefficient used for the TLM branches given the reflection coefficient
"""
function Reflection_coefficient(Γ)
    return ((1+Γ)-(sqrt(2)*(1-Γ))) / ((1+Γ)+(sqrt(2)*(1-Γ)))
end



"""
    time_to_discrete(timeval, fs)

    return the discrete time from the real timeval
"""
function time_to_discrete(timeval::Float64, fs::Int64)
    return Int(timeval * fs)
end




"""
    plot_fft_pressure(pressure1, pressure2, fs, start, stop)

    plot and saves the fft for the given start and stop value
"""
function plot_fft_pressure(pressure1::Vector{Any}, pressure2::Vector{Any}, fs, start, stop)
    start_ind = Int(floor(start * fs)) + 1
    end_ind = Int(floor(stop * fs)) + 1
    
    # Slice the pressure array from start index to the end
    sliced_pressure = pressure1[start_ind:end_ind]
    sliced_pressure2 = pressure2[start_ind:end_ind]




    
    # Convert the pressure arrays to Float64
    sliced_pressure = float.(sliced_pressure)
    sliced_pressure2 = float.(sliced_pressure2)
    
    fft_result = rfft(sliced_pressure .+ eps())
    fft_result2 = rfft(sliced_pressure2 .+ eps())
    
    
    # Compute frequency axis
    N = length(sliced_pressure)
    freq_axis = fs * (0:N÷2) / N
    
    # Convert pressure to decibels (dB)
    pressure_db = 20 * log10.(abs.(fft_result))
    pressure2_db = 20 * log10.(abs.(fft_result2))
    
    # Calculate octave band tick values
    min_freq = 100
    max_freq = fs / 2
    octave_bands = Float64[]
    octave_band = min_freq
    while octave_band <= max_freq
        push!(octave_bands, octave_band)
        octave_band *= 2
    end


    lines = [fs/5.06, fs/3.3]
    # Plot the magnitude spectrum with logarithmic x-axis and octave band tick values
    fig1 = plot(freq_axis, pressure_db, xaxis=:log10, xlabel="Frequency [Hz]", ylabel="Magnitude [dB]", label="Unfiltered FFT", legend=:topleft)
    #plot!(fig1, freq_axis, pressure2_db, label="Lowpass filtered fs/10")
    
    # Manually set the tick values and limits on the x-axis
    plot!(fig1, xticks=(octave_bands, octave_bands), xlims=(min_freq, max_freq))
    vline!([fs/5.06], line=:dash, label="fs/5.06")
    vline!([fs/3.3], line=:dash, label="fs/3.3")

    # Enable minor gridlines on the x-axis with dashed style and alpha value
    plot!(fig1, minorgrid=true, minorgridstyle=:dash, minorgridalpha=0.02)
    display(fig1)
    # Save the plot
    savefig(fig1, string("fft_plot_fs_",fs, ".png"))
end



"""
    find_t60(pressure, pressure2, pressure3, pressure4, time, it, x, y, z)

    takes in 4 different pressure values and calculate the T60 value from an actualy 60 dB drop.
"""
function find_t60(pressure::Vector{Float64},pressure2::Vector{Float64}, pressure3::Vector{Float64}, pressure4::Vector{Float64}, time::Vector{Float64}, it::Int64, X::Float64, Y::Float64, Z::Float64)
    
    # Extrapolate 60 dB drop
    if (impulse == true) it = argmax(pressure3) end
    start_index = it
    end_index = length(pressure3)
    for i in it:length(pressure3)
        if pressure3[i] <= pressure3[it] - 5
            start_index = i
            break
        end
    end

    for i in start_index:length(pressure3)
        if pressure3[i] <= pressure3[it] - 65
            end_index = i
            break
        end
    end

    # Create arrays for the regression
    x = time[start_index:end_index]
    y = pressure3[start_index:end_index]

    # Calculate the regression coefficients
    n = length(x)
    x̄ = mean(x)
    ȳ = mean(y)
    Sxx = sum((x .- x̄) .^ 2)
    Sxy = sum((x .- x̄) .* (y .- ȳ))
    b₁ = Sxy / Sxx
    b₀ = ȳ - b₁ * x̄

    # Calculate the T60 value
    t60 = -60 / b₁
    @printf("T60 value: %.2f s\n", t60)

    V = X * Y * Z
    S = 2 * X * Y + 2 * X * Z + 2 * Y * Z

    D_R = 0.16 * V / (S*(1-R[1]^2))
    @printf("Sabines Equation: %.2f s\n", D_R)
    default(titlefont = font(18, "sans-serif"), guidefont = font(14, "sans-serif"), tickfont = font(14, "sans-serif"), legendfont = font(12, "sans-serif"))

    # Plot the pressure values and regression line
    Plots.plot(time, pressure, xlabel="Time [s]", ylabel="Pressure [dB]", label="Lowpass filtered (source freq + 50 Hz)) Lp")
    Plots.plot!(size=(1000, 800))
    Plots.plot!(time, pressure2, label="Lowpass filtered (fs/6) Lp")
    Plots.plot!(time, pressure3, label="Lowpass filtered (fs/10) Lp")
    Plots.plot!(time, pressure4, label="Unfiltered Lp")
    Plots.plot!(x, b₀ .+ b₁ .* x, label="Regression Line")
    Plots.hline!([pressure3[start_index], pressure3[end_index]], linestyle=:dash, label="Regression bounds, T60")
    
    #ylims!((-50,110))
    xlims!((0.05,0.85))
    
    Plots.savefig(string("TLM NEW R=",R[1], " ",freq, " Hz  fs ",fs , " source ", impulse ? "Impulse" : "Harmonic","T60 ",t60,".png"))
end



"""
    harmonic_val()

    Returns a list containing the harmonic signal of length ´energy_time´, filtered with a tukey window to ensure stability
"""
function harmonic_val()
    t = collect(0:1/fs:energy_time)
    window = tukey(length(t), 0.05)
    return 40*cos.(t*Δt*2*pi*freq) .* window
end



"""
    _real_to_index(val::Array{1,Float64})

    Takes in a 1D array with [x,y,z] position and return the discrete values within the grid
"""
function _real_to_index(val)
    ind = [Int(div(val[1],Δd)), Int(div(val[2],Δd)), Int(div(val[3],Δd))]
    return ind
end



"""
    tlmGrid()

    Create and update the ´Labeled_tlm´ array.
"""
function tlmGrid()
    println("Creating ShoeLabeled_tlm shape")
    println("Δd: ",Δd, " m")
    println("size:\t x_direction: ", nx*Δd, " m \t y_direction: ",ny*Δd," m \t z_direction: ", nz*Δd, " m")

    # Set surface values by checking the position
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz

                if i == 1
                    Labeled_tlm[i, j, k] = 1
                elseif i == nx
                    Labeled_tlm[i, j, k] = 2
                elseif j == 1
                    Labeled_tlm[i, j, k] = 3
                elseif j == ny
                    Labeled_tlm[i, j, k] = 4
                elseif k == 1
                    Labeled_tlm[i, j, k] = 5
                elseif k == nz
                    Labeled_tlm[i, j, k] = 6
                end
            end
        end
    end

    # Set Edge values
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz

                if i == 1 && j == ny
                    Labeled_tlm[i, j, k] = 14
                elseif j == ny && k == 1
                    Labeled_tlm[i, j, k] = 45
                elseif i == nx && j == ny
                    Labeled_tlm[i, j, k] = 24
                elseif j == ny && k == nz
                    Labeled_tlm[i, j, k] = 46
                elseif i == nx && k == 1
                    Labeled_tlm[i, j, k] = 25
                elseif i == nx && k == nz
                    Labeled_tlm[i, j, k] = 26
                elseif i == nx && j == 1
                    Labeled_tlm[i, j, k] = 23
                elseif j == 1 && k == 1
                    Labeled_tlm[i, j, k] = 35
                elseif i == 1 && k == 1
                    Labeled_tlm[i, j, k] = 15
                elseif i == 1 && k == nz
                    Labeled_tlm[i, j, k] = 16
                elseif i == 1 && j == 1
                    Labeled_tlm[i, j, k] = 13
                elseif j == 1 && k == nz
                    Labeled_tlm[i, j, k] = 36
                end
            end
        end
    end

    # Set Corner values
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz

                if i == nx && j == 1 && k == 1
                    Labeled_tlm[i, j, k] = 235
                elseif i == 1 && j == 1 && k == 1
                    Labeled_tlm[i, j, k] = 135
                elseif i == 1 && j == ny && k == 1
                    Labeled_tlm[i, j, k] = 145
                elseif i == nx && j == ny && k == 1
                    Labeled_tlm[i, j, k] = 245
                elseif i == nx && j == 1 && k == nz
                    Labeled_tlm[i, j, k] = 236
                elseif i == 1 && j == 1 && k == nz
                    Labeled_tlm[i, j, k] = 136
                elseif i == 1 && j == ny && k == nz
                    Labeled_tlm[i, j, k] = 146
                elseif i == nx && j == ny && k == nz
                    Labeled_tlm[i, j, k] = 246
                end
            end
        end
    end
    return
end



"""
    calcualate_next_step(it)

    iterate the net timestip with the correct order
"""
function calculate_next_step(it)
    
    # Insert energy into 
    update_incident_with_energy(it)

   

    # Update scatter values
    scatter()
    
    # Update incident pulses
    propagate()
    

end



"""
    meas_pressure()
    
    insert pressure values into the three mic positions as well as the gif_mic, used for animation purposes
"""
function meas_pressure()
    global mic1, mic2, mic3
    for (it, item) in enumerate(pressure_grid)
        push!(mic1, item[_real_to_index(meas_pos1)[1],_real_to_index(meas_pos1)[2],_real_to_index(meas_pos1)[3]])
        push!(mic2, item[_real_to_index(meas_pos2)[1],_real_to_index(meas_pos2)[2],_real_to_index(meas_pos2)[3]])
        push!(mic3, item[_real_to_index(meas_pos3)[1],_real_to_index(meas_pos3)[2],_real_to_index(meas_pos3)[3]])
        


       gif_mic[1:end,1:end,it] = item[1:end,1:end, _real_to_index(source_position)[3]]
    end
end



"""
    animate_heatmap_slices(data, filename)

    Animate a gif out of an array containing heatmaps
"""
function animate_heatmap_slices(data, filename)
    N = size(data, 3)
    prog2 = Progress(N)
    stepsize = 1
    filtered_data = zeros((nx,ny,N))
    for i in 1:nx, j in 1:ny

        filtered_data[i,j,:] = lowpassfilter(data[i, j, :], fs, fs/10)
    end

    data_size = size(filtered_data)

    # Set the plot size based on the heatmap dimensions
    heatmap_size = (data_size[1], data_size[2])

    anim = @gif for i in 1:stepsize:N
        
        heatmap(filtered_data[:, :, i], c=:grays, clim=((-40,10)), title="Slice $i", aspect_ratio = data_size[2] / data_size[1])
        next!(prog2)
    end

    # Save the animation as a file
    gif(anim, filename, fps=15)
end




"""
    run_simulation()

    overal ´main´ code for running the simulation
"""
function run_simulation()
    prog1 = Progress(N)
    println(Δt)
    for it in 1:N
       
        # Caculate values for next timestep
        calculate_next_step(it)

        # Sum overall pressure for each node
        overal_pressure()
        next!(prog1)
        #readline()
    end
    meas_pressure()
    
    #plot_data(ltau(lowpassfilter(mic2,fs,300), fs, Δt * 10, p0), ltau(lowpassfilter(mic2,fs,450), fs, Δt * 10, p0), ltau(lowpassfilter(mic2,fs,600), fs, Δt * 10, p0))
    #ind_source = _real_to_index(source_position, Δd)
    gif_arr = 10*log10.((gif_mic.+ eps()).^2)
    
    (gif_arr, "TEST_GIF.gif")
    ltau_timeconstant = 0.005
    plot_fft_pressure(mic2,lowpassfilter(mic2,fs,fs/10), fs, energy_time, tot_time-0.001)
    find_t60(ltau(lowpassfilter(mic2,fs,freq+50), fs, ltau_timeconstant, p0), ltau(lowpassfilter(mic2,fs,fs/6), fs, ltau_timeconstant, p0) , ltau(lowpassfilter(mic2,fs,fs/10), fs, ltau_timeconstant, p0), ltau(mic2, fs, ltau_timeconstant, p0) ,collect(0:N-1) / fs,time_to_discrete(energy_time,fs)-10, nx*Δd, ny*Δd, nz*Δd)

end



"""
    lowpasfilter(signals, fs, cutoff, order=10)

    Lowpass filter the given array using Butterworth. Can easily be changed to Chebyshev
"""
function lowpassfilter(signals, fs, cutoff, order=10)
    wdo = 2.0 * cutoff / fs
    filth = digitalfilter(Lowpass(wdo), Butterworth(order))  # Increase the roll-off rate by using Chebyshev filter
    filtfilt(filth, signals)
end



"""
    update_incident_with_energy(it)

    Update the array with energy with the chosen energy signal if energy_time states so
"""
function update_incident_with_energy(it)
    if Δt*it >= energy_time

        insert_energy(0)
        return
    end
    if impulse == true
        insert_energy(impulse_value[it])
    end
    if harmonic == true
        insert_energy(harmonic_sig[it])
    end
    if sweep == true
        insert_energy(sweep_sig[it])
    end
end



"""
    insert_energy(val)

    called upon when energy will be inserted
"""
function insert_energy(val)
    source_pos = _real_to_index(source_position)
    val  = val * 10
    IN[source_pos[1]+1,source_pos[2],source_pos[3]] = val
    IE[source_pos[1],source_pos[2]-1,source_pos[3]] = val
    IS[source_pos[1]-1,source_pos[2],source_pos[3]] = val
    IW[source_pos[1],source_pos[2]+1,source_pos[3]] = val
    IU[source_pos[1],source_pos[2],source_pos[3]-1] = val
    ID[source_pos[1],source_pos[2],source_pos[3]+1] = val
end



"""
    overal_pressure()

    Update ´pressure_grid´ with pressure values for the entirety of the simulation. Can be used to process data afterwards
"""
function overal_pressure()
    global pressure_grid
    sum_nodes = ((1/3)*(IN .+ IE .+ IS .+ IW .+ IU .+ ID))
    push!(pressure_grid, sum_nodes)
    return
end



"""
    scatter()

    update the scatter arrays
"""
function scatter()
    global SW, SN, SE, SS, SU, SD
    SW = (1/3) * ((-2*IW) .+ IN .+ IE .+ IS .+ IU .+ ID)
    SN = (1/3) * (IW .+ (-2*IN) .+ IE .+ IS .+ IU .+ ID)
    SE = (1/3) * (IW .+ IN .+ (-2*IE) .+ IS .+ IU .+ ID)
    SS = (1/3) * (IW .+ IN .+ IE .+ (-2*IS) .+ IU .+ ID)
    SU = (1/3) * (IW .+ IN .+ IE .+ IS .+ (-2*IU) .+ ID)
    SD = (1/3) * (IW .+ IN .+ IE .+ IS .+ IU .+ (-2*ID))
    
    return
end



"""
    case(n)

    fetch the lables from the ´Labeled_tlm´ array
"""
function case(n::Int64)
    Diffusor = false
    if n < 0
        Diffusor = true
        n = abs(n)
    end
    num_str = string(n)
    used_nums = [parse(Int, digit) for digit in num_str]
    
    return used_nums, Diffusor
end



"""
    propagate()

    Propagete the scattered pulses to the new incident places
"""
function propagate()
    for i in 1:size(Labeled_tlm,1), j in 1:size(Labeled_tlm,2), k in 1:size(Labeled_tlm,3)
       

        case_used, diffusor = case(Labeled_tlm[i,j,k])
        for n in case_used
            if n == 0 Refl = 0 else Refl = diffusor ? Reflection_coefficient(R[7]) : Reflection_coefficient(R[n]) end
                
            IN[i,j,k] = (1 in case_used) ? Refl * SN[i,j,k] : SS[i - 1, j, k] 
            IS[i,j,k] = (2 in case_used) ? Refl * SS[i,j,k] : SN[i + 1, j, k] 
            IW[i,j,k] = (3 in case_used) ? Refl * SW[i,j,k] : SE[i, j - 1, k] 
            IE[i,j,k] = (4 in case_used) ? Refl * SE[i,j,k] : SW[i, j + 1, k] 
            ID[i,j,k] = (5 in case_used) ? Refl * SD[i,j,k] : SU[i, j, k - 1] 
            IU[i,j,k] = (6 in case_used) ? Refl * SU[i,j,k] : SD[i, j, k + 1]             
                
            
        end
    end
    return
end



"""
    plot_data(arr1, arr2, arr3)

    Plot data for the three arrays used. Generate time axis using ´N´, ensure arrays are of same length as ´N´
"""
function plot_data(arr1, arr2, arr3)
    println("Plotting arrays")
    t = collect(0:N-1) / fs
    
    sos = f.(t)
    #display(arr1)
    Plots.plot(t, arr1, label="Mic 1")
    Plots.plot!(t, arr2, label="Mic 2")
    Plots.plot!(t, arr3, label="Mic 3")
    #Plots.plot!(t, sos, label="600 Hz")
    xlabel!("Time (s)")
    ylabel!("Magnitude (dB)")
    title!("Time weighted sound pressure \nlevel at three positions")
    ylims!((0,120))
    #xlims!(0.25,0.35)
end



"""
    main()

    Initialize the code, creating the grid and running the simulation
"""
function main()
    tlmGrid()
    #Labeled_tlm .= -Labeled_tlm

    if box == true
        boxshape = draw_box_closed([0.1, 0.45 , 0.1], 5, d_box, b_box, h_box, N_QRD, b_QRD, d_max_QRD, d_vegger, d_bakplate, d_skilleplate, d_absorbent_QRD)


        for (i,plane) in enumerate(boxshape)
        
            println("w",i, "\t", plane)
            
            place_wall(Labeled_tlm, plane[1], plane[2], plane[3], Δd)
            
        end


    run_simulation()
end





### Generate the three energy signals 
sweep_sig = generateSweep(50, fs/2-1, energy_time, fs)
harmonic_sig = harmonic_val()
impulse_value = improvised_gaussian()
main()

