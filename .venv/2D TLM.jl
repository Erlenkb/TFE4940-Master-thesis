using FFTW, Plots
using ProgressMeter
using FFMPEG
using DSP
using Statistics, StatsPlots
using PlotThemes
using DelimitedFiles
using Mmap
using SignalAnalysis

#using MeasureIR: expsweep, stimulus, analyze


include("ltau.jl")
include("Draw_box_closed.jl")
include("Place objects into TLM.jl")

### Temp store bigger array in allocated file in main disk
### Optimal for big fs and Time values where "pressure_grid" grow big
#memory_file = "datafile.bin"
#file = open(memory_file, "w")
#write(file, zeros(Float16,(nx,ny,N)))
#close(file)
#pressure_grid = Mmap.mmap(file_path)
#######################################


"""
Parameters above is for the user to edit.
    tot_time: Defines the total simulation time
    energy_time: Defines the time the source is active
    freq: If source type is set to ´haronic´, set the frequency with this parameter
    Length: Set the length of the room
    width: Set the width of the room
    Temp: Define the temperature in Kelvin within the room
    R: Define the reflection coefficient of the diffusor box
    fs: Define the sampling frequeny of the room - Affect the grid distance
"""
tot_time = 5  # in seconds
energy_time = 2  # in seconds
freq = 250
Length = 5
width = 5.0
Temp = 291
R_diffusor = 0.9
R_room = [0.5, 0.5, 0.5, 0.5]
fs = 30000

## Chose the souce signal to be inserted into the array. Only one must be true at the same time
sweep = true 
impulse = false
harmonic = false

box = true   # Set to false if you dont want the box to be placed inside the room

plot_IR = false # Set to true if wanting to plot the IR of the simulated results

source_position = [4.5,2.5]
mic_position1 = [2.5,2.5]
mic_position2 = [0.5,0.5]
mic_position3 = [0.5,0.5]
box_start = [0.25, 2.65]







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

###### Initialize variables ###########
Labeled_tlm = zeros(Int,(nx,ny))
pressure_grid = zeros(Float16,(nx,ny,N))
SN = zeros((nx,ny))
SE = zeros((nx,ny))
SW = zeros((nx,ny))
SS = zeros((nx,ny))
IN = zeros((nx,ny))
IE = zeros((nx,ny))
IS = zeros((nx,ny))
IW = zeros((nx,ny))
positions = Vector{Vector{Int}}(undef, 31)
mic_data = zeros((31,N))
N = Int(tot_time ÷ Δt)
nx = Int(Length ÷ Δd + 1)
ny = Int(width ÷ Δd + 1)
Δd = c / fs
Δt = Δd / c
p0 = 2e-5
ρ_air = 1.225
c = 343.2*sqrt(Temp/293)
theme(:wong2)
#######################################



# Function to create the IR for all measurement positions and store them in respective .txt files to be used in the python
# script that takes the measurement positions and create the polar plot of the diffusive properties.

function store_arrays(data, time)
    # Check if the dimensions of the arrays are compatible
    size(data, 2) == length(time) || error("Mismatch in array dimensions")
    
    directory = "Simu_files_fs_40khz"
    
    if !isdir(directory)
        mkdir(directory)
    end

    for i in 1:31
        IR, time_IR = find_impulse_response_sweep(sweep_sig,data[i, 1:end],fs)
        array_data = hcat(time_IR, IR)
        filename = "$(directory)/1_$(i)_S01_R01.txt"   # Set to 2_${i}_S01_R01 when using the ref. arrays
        # Save array data to a text file
        writedlm(filename, array_data, '\t')
        println("Array $i saved to $filename")
    end
end



# Takes in the source signal and the measured signal for the given measurement position and return the IR.

function find_impulse_response_sweep(sweep_signal, measured_signal, fs)
    # Normalize the signals
    sweep_signal /= rms(sweep_signal)
    measured_signal /= rms(lowpassfilter(measured_signal, fs, fs/10))

    # Add a small constant to the first element of the sweep signal
    sweep_signal[1] += eps()

    # Perform deconvolution using least squares estimation
    impulse_response = conv(measured_signal, sweep_signal)

    # Calculate time axis for impulse response
    time_axis = (0:length(impulse_response)-1) / fs

    if plot_IR == true
        display(Plots.plot(time_axis, impulse_response))
    end

    return impulse_response, time_axis
end



# Takes in the start position as an array, a radius value and a step value. It generate positions where each microphone is placed
# Update the global microphone position Array

function generate_half_circle(position::Array{Float64,1},radius::Float64, theta::Float64)

    x,y = position
    i = 1
    for angle in -5π/12:theta:(5π/12+theta)
        new_x = x + radius * cos(angle)
        new_y = y + radius * sin(angle)
        positions[i] = _real_to_index([new_x, new_y])
        i += 1
    end
    return
end


# Takes it the iteration value for the main ´for´ loop and the array containing the pressure values for the given timestep.
# It then updates the microphone data arrays containing live measurements.

function update_meas_mics(it, arr)
    for i in 1:31
        pos = positions[i]
        x,y = pos[1], pos[2]
        mic_data[i, it] = arr[x,y]
    end
end

# Plot the given three arrays

function plot_arrays(arr1::Array, arr2::Array, arr3::Array)
    println("Plotting arrays")
    t = collect(0:length(arr1)-1) / fs
    
    fig = Plots.plot(t, arr1, label="Mic @source (dB)")
    fig = Plots.plot!(t, arr2, label="Mic 1 m from source (dB)")
    fig = Plots.plot!(t, arr3, label="Mic 2 m from source (dB)")
    xlabel!("Time (s)")
    ylabel!("Magnitude (dB)")
    title!("Time weighted sound pressure \nlevel at three positions")
    ylims!((0,105))
    println("saving figure: ", string("R=",R, " fs ",fs ," source sweep size ",round(nx*Δd,digits=1),"x",round(ny*Δd,digits=1),".png"))
    
    Plots.savefig(fig, string("R=",R, " fs ",fs ," source sweep size ",round(nx*Δd,digits=1),"x",round(ny*Δd,digits=1),".png"))
end

# Plot the FFT pressure for two arrays for a given start and stop value.

function plot_fft_pressure(pressure1, pressure2, fs, start, stop, num)
    start_ind = Int(start * fs) + 1
    end_ind = Int(stop * fs) + 1
    
    # Slice the pressure array from start index to the end
    sliced_pressure = pressure1[start_ind:end_ind]
    sliced_pressure2 = pressure2[start_ind:end_ind]
    
    fft_result = rfft(sliced_pressure)
    fft_result2 = rfft(sliced_pressure2)
    
    # Compute frequency axis
       
    N = length(sliced_pressure)
    freq_axis = fs * (0:N÷2) / N
    # Compute the FFT using rfft
    
    # Convert pressure to decibels (dB)
    pressure_db = 20 * log10.(abs.(fft_result))
    pressure2_db = 20*log10.(abs.(fft_result2))

    # Calculate octave band tick values
    min_freq = 100
    max_freq = fs / 2
    octave_bands = Float64[]
    octave_band = min_freq
    while octave_band <= max_freq
        push!(octave_bands, octave_band)
        octave_band *= 2
    end


    # Find the frequency with the highest power
    max_power_idx = argmax(pressure_db)
    max_power_freq = freq_axis[max_power_idx]

    indices = sortperm(pressure_db, rev=true)
    top10Magnitudes = freq_axis[indices[1:10]]

    println("Top 10 FFT Magnitudes:")
    for (index, magnitude) in enumerate(top10Magnitudes)
        println("Freq $index: $magnitude")
    end




    # Print the frequency with the highest power
    println("Frequency with the highest power: ", round(max_power_freq,digits=1), " Hz")

    
    # Plot the magnitude spectrum with logarithmic x-axis and octave band tick values
    fig1 = plot(freq_axis, pressure_db, xaxis=:log10, xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", legend=false, xlim=(100, fs/5),
         xscale=:log10)#, xticks=(octave_bands, octave_bands))
    fig1 = plot!(freq_axis, pressure2_db, xaxis=:log10, xlim=(100, fs/5),xscale=:log10)
    
    # Enable minor gridlines on the x-axis with dashed style and alpha value
    fig1 = plot!(minorgrid=true, minorgridstyle=:dash, minorgridalpha=0.02)
    println("saving figure: ", string("FFT R=",R, " fs ",fs ," source sweep size ",round(nx*Δd,digits=1),"x",round(ny*Δd,digits=1),".png"))
    # Save the plot
    Plots.savefig(fig1, string(num,"FFT R=",R, " fs ",fs ," source sweep size ",round(nx*Δd,digits=1),"x",round(ny*Δd,digits=1),".png"))
end

function animate_heatmap_slices(data, filename)
    N = size(data, 3)
    stepsize = 1
    prog_anim = Progress(N)
    anim = @gif for i in 1:stepsize:N
        heatmap(data[:, :, i], c=:grays, clim=(-40,10), title="Slice $i")
        next!(prog_anim)
    end

    # Save the animation as a file
    gif(anim, filename, fps=50)
end

function harmonic_val(it)
    return 40*cos(it*Δt*2*pi*freq)
end

function create_shoebox()

    println("Creating box")
    println("Δd: ",Δd, " m")
    println("size:\t x_direction: ", nx*Δd, " m \t y_direction: ",ny*Δd," m")
    
    Labeled_tlm[2:end-1,1] .= 3
    Labeled_tlm[2:end-1,end] .= 4
    Labeled_tlm[1,2:end-1] .= -1
    Labeled_tlm[end, 2:end-1] .= -2
    Labeled_tlm[1,1] = -13
    Labeled_tlm[1,end] = -14
    Labeled_tlm[end,1] = -23
    Labeled_tlm[end,end] = -24
    return 
end

function generateSweep(start_freq, end_freq, duration, sample_rate)
    t = collect(0:1/sample_rate:duration)
    freq_range = end_freq - start_freq
    phase = 2π * (start_freq * t + 0.5 * freq_range * t.^2 / duration)
    sweep_signal = sin.(phase)
    
    window = tukey(length(sweep_signal), 0.05)  
    sweep_signal = sweep_signal .* window
    
    return sweep_signal
end

function propagate()
    for i in 1:nx, j in 1:ny
        case_used, negative = case(Labeled_tlm[i,j])
        for n in case_used

            if n == 0 Refl = 0 else Refl = diffusor ? Reflection_coefficient(R_diffusor) : Reflection_coefficient(R_room,[n]) end
            IN[i,j] = (1 in case_used) ? Refl * SN[i,j] : SS[i - 1, j]
            IS[i,j] = (2 in case_used) ? Refl * SS[i,j] : SN[i + 1, j]
            IW[i,j] = (3 in case_used) ? Refl * SW[i,j] : SE[i,j  - 1]
            IE[i,j] = (4 in case_used) ? Refl * SE[i,j] : SW[i, j + 1]
        end
    end
    return
end

function scatter()

    global SW = (1/2) * (-IW .+ IN .+ IE .+ IS)
    global SN = (1/2) * (IW .- IN .+ IE .+ IS)
    global SE = (1/2) * (IW .+ IN .- IE .+ IS)
    global SS = (1/2) * (IW .+ IN .+ IE .- IS)
    return
end

function overal_pressure()
    return (1/2) .*(IN .+ IE .+ IS .+ IW)
end

function case(n::Int64)
    negative = false
    if n < 0
        negative = true
        n = abs(n)
    end
    num_str = string(n)
    return [parse(Int, digit) for digit in num_str], negative
end

function _real_to_index(val::Array{Float64,1})
    ind = [Int(div(val[1],Δd)), Int(div(val[2],Δd))]
    return ind
end

function update_incident_with_energy(it)
    if Δt*it >= energy_time
        return 
    end

    if sweep == true
        insert_energy(sweep_sig[it])
    end

    if impulse == true
        insert_energy(gaussian_pulse(it, Int(energy_time÷Δt),-5,4))
    end
    if harmonic == true
        insert_energy(harmonic_val(it))
    end
    return 
end

function insert_energy(val)
    source_pos = _real_to_index(source_position)
    val  = val * 10
    #IN[source_pos[1]+1,source_pos[2]] = val
    #IE[source_pos[1],source_pos[2]-1] = val
    IS[source_pos[1]-1,source_pos[2]] = val
    #IW[source_pos[1],source_pos[2]+1] = val
    return 
end


function lowpassfilter(signals, fs, cutoff, order=4)
    wdo = 2.0 * cutoff / fs
    filth = digitalfilter(Lowpass(wdo), Butterworth(order))
    filtfilt(filth, signals)
end



function lowpass_func(fs, cutoff, f_order, pressure)

    # Assuming you have a pressure array called "pressure" with sampling frequency 44100 Hz
    sampling_freq = fs
    cutoff_freq = cutoff

    # Design a lowpass filter
    filter_order = f_order
    lpf = Lowpass(cutoff_freq, filter_order, sampling_freq)

    # Apply the filter to the pressure array
    filtered_pressure = filt(lpf, pressure)
    return filtered_pressure
end

function run_simulation()
    prog1 = Progress(N)

    mic1 = zeros(N)
    mic2 = zeros(N)
    mic3 = zeros(N)
    ind1 = _real_to_index(mic_position1)
    ind2 = _real_to_index(mic_position2)
    ind3 = _real_to_index(mic_position3)

    generate_half_circle([0.5, 2.5], 2.1, 0.09)
    time = collect(range(0, stop=(N-1)*Δt, step=Δt))
    time_sweep = collect(range(0, stop=(length(sweep_sig)-1)*Δt, step=Δt))


    
    for it in 1:N
        # Caculate values for next timestep
        calculate_next_step(it)

        # Sum overall pressure for each node
        pressure_grid[:,:,it] = overal_pressure()
        update_meas_mics(it, overal_pressure())
        mic1[it] = overal_pressure()[ind1[1], ind1[2]]

        next!(prog1)    
        #readline()
    end
    update_meas_mics()

    #### Unomment commands below if needed for other 
    #mic1_fft = pressure_grid[ind1[1],ind1[2], 1:end]
    #mic1 = ltau(pressure_grid[ind1[1], ind1[2],1:end], fs, 0.01, 2e-5)
    #mic2 = ltau(pressure_grid[ind2[1], ind2[2],1:end], fs, 0.01, 2e-5)
    #mic3 = ltau(pressure_grid[ind3[1], ind3[2],1:end], fs, 0.01, 2e-5)
    #plot_arrays(mic1, mic2, mic3)
    #plot_fft_pressure(mic1_fft, fs, energy_time)
    store_arrays(mic_data, time)
    
    #animate_heatmap_slices(10*log10.(abs.(pressure_grid)),"2D TLM.gif")
end


function gaussian_pulse(timestep, peak_time, sigma)
    amplitude = 10.0  # Amplitude of the Gaussian pulse
    mean = peak_time  # Mean (center) of the Gaussian pulse

    # Calculate the value of the Gaussian pulse at the given timestep
    value = amplitude * exp(-((timestep - mean) ^ 2) / (2 * sigma ^ 2))

    return value
end


function calculate_next_step(it)
    
    # Insert energy into 
    update_incident_with_energy(it)

    # Update scatter values
    scatter()
    
    # Update incident pulses
    propagate()
    
    return
end




function Merge(A::Int, B::Int)::Int
    # Concatenate A and B as strings
    str = string(A, B)

    # Remove duplicates and sort the remaining characters
    unique_chars = sort(unique(str))

    # Convert the sorted string back to an integer and return it
    result = 0
    for c in unique_chars
        result = result * 10 + (c - '0')
    end
    return result
end





create_shoebox()


if box == true
    boxshape = draw_box_closed_2D(box_start, d_box, b_box, N_QRD, b_QRD, d_max_QRD, d_vegger, d_bakplate, d_skilleplate, d_absorbent_QRD)


    for (i,plane) in enumerate(boxshape)
        println("w",i, "\t", plane)
        place_wall2D(Labeled_tlm, plane[1], plane[2], Δd)
    end
end

sweep_sig = generateSweep(100, fs / 2, energy_time, fs)


run_simulation()


