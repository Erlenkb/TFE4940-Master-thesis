#using PlotlyJS
using Plots
using Printf
using DSP
using Statistics, StatsPlots
using FileIO, VideoIO
using ProgressMeter
using Base: log
using FileIO
using CSV, DataFrames
using NPZ
using FFMPEG
using SignalAnalysis


#using Polynomials
#using ControlSystems
"""
using LinearAlgebra
using Plots
using Statistics, StatsPlots
using Polynomials
using Printf
using PlotlyJS
"""

"""
using GLM --> Look into that

plot source and then 1m

Oria - NTNU bibliotek

27.april next meeting
"""

include("Draw_box_closed.jl")
include("ltau.jl")
include("TLM operations.jl")
include("Place objects into TLM.jl")
include("TLM creation.jl")
include("Acoustical operations.jl")
include("TLM calculation.jl")
#include("TLM calculation Guillaume method.jl")
#include("test.jl")
include("Plot_heatmap.jl")



"""
GMSH - Tool for Julia - Mesh generator
Sketchup - Draw grid/walls


--> Check slice in other dimension

Time weighted L_fast 
"""


# ISO 17497 -- Scattering standards


###Gaussioan pulse and measure frequency dependent reverberation time

####### GLOBAL PARAMETERS ########
Temp = 291
ρ_air = 1.225
c = 343.2*sqrt(Temp/293)

freq = 100
fs = 4000
Time = 10.0
po = 2e-5
p0 = 2e-5

Z_T = c *ρ_air
Z_a_concrete = 4400 * 2400
Z_a_concrete = 9500000
Z_a_wood = 2200500
#R = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
#R = [Z_a_concrete, Z_a_concrete, Z_a_concrete, Z_a_concrete, Z_a_concrete, Z_a_concrete, Z_a_concrete] # The characteristic impedance for the different walls in the room
#R = [Z_a_wood, Z_a_wood, Z_a_wood, Z_a_wood, Z_a_wood, Z_a_wood, Z_a_wood]
#R = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

R = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]
#R = [Z_T, Z_T, Z_T, Z_T, Z_T, Z_T, Z_T]
#R = [0,0,0,0,0,0]

# What type of source should be used. priority directional -> harmonic -> impulse
impulse = true
harmonic = false
harmonic_directional = false

# Positions for the different sources and microphone positions. Given in meters
harm_pos = [1.5,1.5,1.5]
imp_pos = [1.5, 1.5, 1.5]
meas_position1 = [1.0,1.0,1.2]
meas_position2 = [2.5,2.5,1.2]
meas_position3 = [1.9,1.9,1.2]

# Strength value for the impulse value and harmonic source as well as the total time the source should be on
# Strength given in Pascal
imp_val_p = 10
signal_strength_Pa = 400
time_source = 5.0




######################################

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


# Tuckey window for energy signal
# macgyver fix hanning windows


function plot_fft_pressure(pressure, fs, start)
    start_ind = Int(start * fs) + 1
    end_ind = length(pressure)
    
    # Slice the pressure array from start index to the end
    sliced_pressure = pressure[start_ind:end_ind]
    
    fft_result = rfft(sliced_pressure)
    
    # Compute frequency axis
       
    N = length(sliced_pressure)
    freq_axis = fs * (0:N÷2) / N
    # Compute the FFT using rfft
    
    # Convert pressure to decibels (dB)
    pressure_db = 20 * log10.(abs.(fft_result))

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
    fig1 = plot(freq_axis, pressure_db, xaxis=:log10, xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", legend=false, xlim=(100, fs/2),
         xscale=:log10)#, xticks=(octave_bands, octave_bands))
    
    # Enable minor gridlines on the x-axis with dashed style and alpha value
    fig1 = plot!(minorgrid=true, minorgridstyle=:dash, minorgridalpha=0.05)
    println("saving figure: ", string("FFT R=",R, " fs ",fs ," source sweep size ",round(nx*Δd,digits=1),"x",round(ny*Δd,digits=1),".png"))
    # Save the plot
    Plots.savefig(fig1, string("FFT R=",R, " fs ",fs ," source sweep size ",round(nx*Δd,digits=1),"x",round(ny*Δd,digits=1),".png"))
end







function TLM(Δd, height, width, length)
    nx = length ÷ Δd
    ny = width ÷ Δd
    nz = height ÷ Δd
    tlm = zeros((nx,ny,nz))
    return tlm
end


function find_t60(pressure::Vector{Float64}, time::Vector{Float64}, it::Int64, X::Float64, Y::Float64, Z::Float64)
    # Find the start and end indices for the regression interval

    # Change to extrapolate to 60 dB drop
    if (impulse == true) it = argmax(pressure) end
    start_index = it
    end_index = length(pressure)
    for i in it:length(pressure)
        if pressure[i] <= pressure[it] - 0.1
            start_index = i
            break
        end
    end
    for i in start_index:length(pressure)
        if pressure[i] <= pressure[it] - 60.1
            end_index = i
            break
        end
    end

    # Create arrays for the regression
    x = time[start_index:end_index]
    y = pressure[start_index:end_index]

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

    D_R = 0.16 * V / (S*(1-R[1]))
    @printf("Sabines Equation: %.2f s\n", D_R)

    # Plot the pressure values and regression line
    Plots.plot(time, pressure, xlabel="Time (s)", ylabel="Pressure (dB)", label="Pressure")
    Plots.plot!(x, b₀ .+ b₁ .* x, label="Regression Line")
    Plots.hline!([pressure[start_index], pressure[end_index]], linestyle=:dash, label="Regression bounds, T60")
    #ylims!((0,120))
    Plots.savefig(string("R=",R[1], " ",freq, " Hz  fs ",fs , " source ", impulse ? "Impulse" : "Harmonic","T60 ",t60,".png"))
end


function generate_half_circle(position::Array{Float64,1}, radius::Float64, theta::Float64)
    positions = Vector{Vector{Int}}(undef, 31)
    
    x, y, z = position
    i = 1
    for angle in -5π/12:theta:5π/12
        new_x = x + radius * cos(angle)
        new_y = y + radius * sin(angle)
        positions[i] = _real_to_index([new_x, new_y, z],Δd)
        i += 1
    end
    return positions
end


function update_measuring_mics(it, positions, pressure_grid::Array{Float64,3}, mic_data)
    for i in 1:30
        pos = positions[i]
        x, y, z = pos[1], pos[2], pos[3]
        mic_data[i, it] = pressure_grid[x,y,z]
    end
end



function generate_half_circle_and_update_values(it, position::Array{Float64,1}, pressure_grid::Array{Float64,3}, radius::Float64, theta::Float64)
    positions = Vector{Vector{Int}}(undef, 31)
    
    x, y, z = position
    i = 1
    for angle in -5π/12:theta:5π/12
        new_x = x + radius * cos(angle)
        new_y = y + radius * sin(angle)
        positions[i] = _real_to_index([new_x, new_y, z],Δd)
        i += 1
    end

    mic1[it] = pressure_grid[positions[1][1], positions[1][2], positions[1][3]]
    mic2[it] = pressure_grid[positions[2][1], positions[2][2], positions[2][3]]
    mic3[it] = pressure_grid[positions[3][1], positions[3][2], positions[3][3]]
    mic4[it] = pressure_grid[positions[4][1], positions[4][2], positions[4][3]]
    mic5[it] = pressure_grid[positions[5][1], positions[5][2], positions[5][3]]
    mic6[it] = pressure_grid[positions[6][1], positions[6][2], positions[6][3]]
    mic7[it] = pressure_grid[positions[7][1], positions[7][2], positions[7][3]]
    mic8[it] = pressure_grid[positions[8][1], positions[8][2], positions[8][3]]
    mic9[it] = pressure_grid[positions[9][1], positions[9][2], positions[9][3]]
    mic10[it] = pressure_grid[positions[10][1], positions[10][2], positions[10][3]]
    mic11[it] = pressure_grid[positions[11][1], positions[11][2], positions[11][3]]
    mic12[it] = pressure_grid[positions[12][1], positions[12][2], positions[12][3]]
    mic13[it] = pressure_grid[positions[13][1], positions[13][2], positions[13][3]]
    mic14[it] = pressure_grid[positions[14][1], positions[14][2], positions[14][3]]
    mic15[it] = pressure_grid[positions[15][1], positions[15][2], positions[15][3]]
    mic16[it] = pressure_grid[positions[16][1], positions[16][2], positions[16][3]]
    mic17[it] = pressure_grid[positions[17][1], positions[17][2], positions[17][3]]
    mic18[it] = pressure_grid[positions[18][1], positions[18][2], positions[18][3]]
    mic19[it] = pressure_grid[positions[19][1], positions[19][2], positions[19][3]]
    mic20[it] = pressure_grid[positions[20][1], positions[20][2], positions[20][3]]
    mic21[it] = pressure_grid[positions[21][1], positions[21][2], positions[21][3]]
    mic22[it] = pressure_grid[positions[22][1], positions[22][2], positions[22][3]]
    mic23[it] = pressure_grid[positions[23][1], positions[23][2], positions[23][3]]
    mic24[it] = pressure_grid[positions[24][1], positions[24][2], positions[24][3]]
    mic25[it] = pressure_grid[positions[25][1], positions[25][2], positions[25][3]]
    mic26[it] = pressure_grid[positions[26][1], positions[26][2], positions[26][3]]
    mic27[it] = pressure_grid[positions[27][1], positions[27][2], positions[27][3]]
    mic28[it] = pressure_grid[positions[28][1], positions[28][2], positions[28][3]]
    mic29[it] = pressure_grid[positions[29][1], positions[29][2], positions[29][3]]
    mic30[it] = pressure_grid[positions[30][1], positions[30][2], positions[30][3]]
    #mic31[it] = pressure_grid[positions[31][1], positions[31][2], positions[31][3]]

    return positions

end





function plot_arrays(arr1::Array, arr2::Array, arr3::Array, Δt::Float64)
    println("Plotting arrays")
    t = collect(0:length(arr1)-1) / fs
    
    Plots.plot(t, arr1, label="Mic @source (dB)")
    Plots.plot!(t, arr2, label="Mic 1 m from source (dB)")
    Plots.plot!(t, arr3, label="Mic 2 m from source (dB)")
    xlabel!("Time (s)")
    ylabel!("Magnitude (dB)")
    title!("Time weighted sound pressure \nlevel at three positions")
    ylims!((25,132))
    Plots.savefig(string("R=",R[1], " ",freq, " Hz  fs ",fs ," source ", impulse ? "Impulse" : "Harmonic",".png"))
end


function _time_to_samples(time::Float64, Δt::Float64)
    return Int(time÷Δt)
end



function iterate_grid(T::Float64, Δd, pressure_grid::Array{Float64,3}, SN::Array{Float64,3}, SE::Array{Float64,3}, SS::Array{Float64,3}, SW::Array{Float64,3}, SU::Array{Float64,3}, SD::Array{Float64,3},IN::Array{Float64,3}, IE::Array{Float64,3}, IS::Array{Float64,3}, IW::Array{Float64,3}, IU::Array{Float64,3}, ID::Array{Float64,3})
    
    # Create time step value, total iteration number N and the sizes of the matrix
    
    println("fs: ",fs)
    L = size(pressure_grid,1)
    M = size(pressure_grid,2)
    N_length = size(pressure_grid,3)

    println("L: ", L,"\tM ", M, "\tN ", N_length)
    println("Time interval between each step: ", 1000*Δt, " ms")
    println("The tot iteration number is: ", N)

    # Generate white noise with the sound level, in dB, given by ´sound_level´
    sound_level = 144.0
    WHITE_NOISE = generate_noise_signal(N, Δt, sound_level)


    meas_positions = generate_half_circle([0.75, 2.5, 2.1], 2.1, 0.09)
    #sweep, sweep_length = logarithmic_sweep(30,1000,3,fs)


    # Create arrays for data collection for plots and heatmaps.
    p_arr = Float64[]
    meas_pos1 = []
    meas_pos2 = []
    meas_pos3 = []
    gif_arr_x = []
    gif_arr_y = []
    gif_arr_z = []
    press_arr = []

    # Create coordinates for the microphone position. Turn real coordinates into indicies
    ind1 = _real_to_index(meas_position1, Δd)
    ind2 = _real_to_index(meas_position2, Δd)
    ind3 = _real_to_index(meas_position3, Δd)

    IN_sum = []
    IE_sum = []
    IS_sum = []
    IW_sum = []
    IU_sum = []
    ID_sum = []

    feedback = zeros(N)


    sweep_signal = generateSweep(100, 10000, time_source, fs)
    #if N < length(sweep_signal) feedback[1:N] = abs.(sweep_signal[1:N])
    #else feedback[1:length(sweep_signal)] = abs.(sweep_signal) end

    # Track the progress using progressmeter to better visualize ETA
    prog1 = Progress(N)

    
    ######## Step 1 - Insert energy into grid

    # Iterate through the matrix with timesteps "n" for a given total time "T"
    for n in 1:N
        #println("Step: ", n)
        # Step 1 - Insert energy into grid
        
        # Use the global boolean parameter ´impulse´ if the initial scattering matrix will have an impulse signal
        if (impulse == true) && (n <= length(sweep_signal))
            #println("Inserting Impulse")
            val = 10
            i = _real_to_index(imp_pos,Δd)[1]
            j = _real_to_index(imp_pos,Δd)[2]
            k = _real_to_index(imp_pos,Δd)[3]
            IN[i,j,k] = val * sweep_signal[n]
            IE[i,j,k] = val * sweep_signal[n]
            IS[i,j,k] = val * sweep_signal[n]
            IW[i,j,k] = val * sweep_signal[n]
            IU[i,j,k] = val * sweep_signal[n]
            ID[i,j,k] = val * sweep_signal[n]
            
        end
       
        # Use the global boolean parameter ´harmonic´ if the grid should experience a harmonic time signal - Will work as a point source located at a single node 
        if (harmonic == true) && (n <= _time_to_samples(time_source,Δt))

            if n == 1
                println("Insert sweep")
                #println("Time_to_samples for ",time_source, " s - ", _time_to_samples(time_source,Δt))
            end

            #A_val = 10* sweep[n]
            # Locate the discrete indicies from the real positions
            i = _real_to_index(harm_pos,Δd)[1]
            j = _real_to_index(harm_pos,Δd)[2]
            k = _real_to_index(harm_pos,Δd)[3]
            IN[i,j,k] = pressure_value(n*Δt)
            IE[i,j,k] = pressure_value(n*Δt)
            IS[i,j,k] = pressure_value(n*Δt)
            IW[i,j,k] = pressure_value(n*Δt)
            IU[i,j,k] = pressure_value(n*Δt)
            ID[i,j,k] = pressure_value(n*Δt)
        end

        # Use the global boolean parameter harmonic_directional if the grid should experience a harmoncic time signal in only one direction / Weaker in some directions - Will be more general in the future.
        if harmonic_directional == true
            # Locate the discrete indicies from the real positions
            i = harm_pos[1]
            j = harm_pos[2]
            k = harm_pos[3]
            IN[i,j,k] = pressure_value_directional(n*Δt)[1]
            IE[i,j,k] = pressure_value_directional(n*Δt)[2]
            IS[i,j,k] = pressure_value_directional(n*Δt)[3]
            IW[i,j,k] = pressure_value_directional(n*Δt)[4]
            IU[i,j,k] = pressure_value_directional(n*Δt)[5]
            ID[i,j,k] = pressure_value_directional(n*Δt)[6]
        end
        


        ######## Step 2 - Calculate overall pressure
        pressure_grid = overal_pressure(IN, IE, IS, IW, IU, ID)


        #display(pressure_grid)

        #push!(p_arr, sum(pressure_grid))
        """
        push!(IN_sum, sum(IN))
        push!(IE_sum, sum(IE))
        push!(IS_sum, sum(IS))
        push!(IW_sum, sum(IW))
        push!(ID_sum, sum(ID))
        push!(IU_sum, sum(IU))

        ######## Step 2.1 - Insert values into different arrays for plots and animation
        """
        fft_mic[n] = pressure_grid[ind1[1],ind1[2],ind1[3]]
        meas1 = pressure_grid[ind1[1],ind1[2],ind1[3]]
        meas2 = pressure_grid[ind2[1],ind2[2],ind2[3]]
        meas3 = pressure_grid[ind3[1],ind3[2],ind3[3]]
        """
        push!(gif_arr_x, 10*log10.(pressure_grid[ind1[1],:,:].^2))
        push!(gif_arr_y, 10*log10.(pressure_grid[:,ind1[2],:].^2))
        
        push!(press_arr, 10*log10.(pressure_grid[ind2[1],ind2[2],ind2[3]].^2))

        #display(10*log10.(pressure_grid[:,:,ind1[3]].^2))

        """
        push!(meas_pos1, meas1)
        push!(meas_pos2, meas2)
        push!(meas_pos3, meas3)

        #push!(p_arr, sqrt((1/(L*M*N_length))*sum(pressure_grid.^2)))
        
        
        # Store mic data for the 31 microphone positions needed for ISO:17497-2:2012
        #generate_half_circle_and_update_values(n, [0.75, 2.5, 2.1], pressure_grid, 2.1, 0.09)
        #update_measuring_mics(n, meas_positions, pressure_grid, mic_array)
        #push!(gif_arr_z, 10*log10.(pressure_grid[:,:,ind1[3]].^2))



        ######## Step 3 - create the scattering matrix
        SN, SE, SS, SW, SU, SD = scattering(IN, IE, IS, IW, IU, ID)





        """
         ####### Checking for places where energy is NOT conserved
         Conservation_checker_array = (IN .- SN) .+ (IE .- SE) .+ (IS .- SS) .+ (IW .- SW) .+ (IU .- SU) .+ (ID .- SD) 
         display(Conservation_checker_array)
         
         positions = findall(x -> x != 0.0, Conservation_checker_array)
         
         println("Harmonic source position: ", _real_to_index(harm_pos,Δd))
         for pos in positions
             i,j,k = pos.I
             println(pos)
             println("North: ", IN[i,j,k] - SN[i,j,k])
             println("East: ", IE[i,j,k] - SE[i,j,k])
             println("South: ", IS[i,j,k] - SS[i,j,k])
             println("West: ", IW[i,j,k] - SW[i,j,k])
             println("Up: ", IU[i,j,k] - SU[i,j,k])
             println("Down: ", ID[i,j,k] - SD[i,j,k])

             println("SUM: ", (IN[i,j,k] - SN[i,j,k]) + (IE[i,j,k] - SE[i,j,k]) + (IS[i,j,k] - SS[i,j,k]) + (IW[i,j,k] - SW[i,j,k]) + (IU[i,j,k] - SU[i,j,k]) + (ID[i,j,k] - SD[i,j,k]))
         end
         """


        ######## Step 4 - create the Incident matrix
        IN, IE, IS, IW, IU, ID = propagate2(Labeled_tlm, SN, SE, SS, SW, SU, SD)
        
       

        # next iteraton for the progressmeter bar
        
        next!(prog1)
        #readline()
    end

    # Indicate that propagation is done
    println("Finished with propagation")


    # Create a time vector ´x´ for the total discrete time ´N´
    x = collect(range(0, stop=(N-1)*Δt, step=Δt))

    # Create the time weigthed sound power level using the ´ltau´ function from Guillaume D.
    #p1 = ltau(lowpassfilter(meas_pos1, fs, 400), fs, 10*Δt, p0)
    #p1_2 = ltau(lowpassfilter(meas_pos2, fs, 400), fs, 10*Δt, p0)
    #p1_3 = ltau(lowpassfilter(meas_pos3, fs, 400), fs, 10*Δt, p0)
    #p1 = ltau(meas_pos1, fs, 10*Δt, p0)
    #p1_2 = ltau(meas_pos2, fs, 10*Δt, p0)
    #p1_3 = ltau(meas_pos3, fs, 10*Δt, p0)
    #press_arr = replace(press_arr, NaN => 0.0)
    #press_arr = convert(Vector{Float64}, map(x -> parse(Float64, x), press_arr))


    # Create a pressure array using the ´_Lp´ function
    #p_arr = _Lp(p_arr)
    #Plots.plot(x,press_arr)
    #Plots.plot(x, press_arr, xlabel="Time (s)", ylabel="Pressure", label="Overall Pressure", title=string("Total energy with R=1\nmax val: ", @sprintf("%.3e",maximum(p_arr)), "     min val: ", @sprintf("%.3e",minimum(p_arr))), yaxis=:log10)
    #Plots.plot(x, IN_sum.-IS_sum, label="IN_sum")
    #Plots.plot!(x, IE_sum.-IW_sum, label="IE_sum")
    #Plots.plot!(x, p_arr, label="Overall pressure")
    #Plots.plot!(x, abs.(IS_sum), label="IS_sum")
    #Plots.plot!(x, abs.(IW_sum), label="IW_sum")
    #Plots.plot!(x, IU_sum .- ID_sum, label="IU_sum")
    #Plots.plot!(x, abs.(ID_sum), label="ID_sum")

    #Plots.savefig("sum pressure all nodes.png")
    #Plots.plot(x, 10*log10.((p_arr.^2) / p0^2))
    #println("max val: ", maximum(p_arr), "\tmin val: ", minimum(p_arr))

    #Plots.savefig("Impulse response free field.png")

    plot_fft_pressure(fft_mic, fs, time_source)
   
    # Different actions to be done on the finished result. remve comment mark and edit the input parameters to alter the functions.
    #find_t60(p1_2, x, _time_to_samples(time_source,Δt)-10, L*Δd, M*Δd, N_length*Δd)
    
    #plot_arrays(p_arr, p1_2, p1_3, Δt)
    #_heatmap_gif11(gif_arr_x, N,x, Δd, string("R=",R[1], " ",freq, " Hz  fs ",fs , " ",T," s source ", impulse ? "Impulse" : "Harmonic" ," x plane.gif"))
    #_heatmap_gif11(gif_arr_y, N,x, Δd, string("R=",R[1], " ",freq, " Hz  fs ",fs , " ",T," s source ", impulse ? "Impulse" : "Harmonic" ," y plane.gif"))
    
    """
    L = size(gif_arr_z[1],1)
    M = size(gif_arr_z[1],2)

    new_gif_arr = zeros((L,M,N))
    for (i, val) in enumerate(gif_arr_z)
        for x in range(1,L)
            for j in range(1,M)
                new_gif_arr[x,j,i] = val[x,j]
            end
        end
    end
    nx, ny, nz = size(new_gif_arr)
    println("nx: ", nx, "\tny: ", ny, "\tnz: ", nz)
    animate_heatmap_slices(new_gif_arr, string("R=",R[1], " ",freq, " Hz  fs ",fs , " ",T," s source ", impulse ? "Impulse" : "Harmonic" ," z plane.gif"))
    """
    
end







###### Main code 

# create the distance value between each node in the TLM matrix

# Create 31 microphone arrays used to store data






# Create the pressure- and scattering arrays as well as the labaled arrays and the pressure grid.




# Create the shape of the box given the start positions and the global box parameters given in the top of the file - The box
# is created by starting from the bottom left of the box and goes clockwise depending on the position value used, that define the surface to place the box to.



#place_wall(Labeled_tlm, [0.5, 3.0], [1.0, 1.0], [0.001, 2.5], Δd) # Y plane
#place_wall(Labeled_tlm, [0.5, 3.0], [0.5, 3.0], [1.0, 1.0], Δd) # Z plane 
#place_wall(Labeled_tlm, [1.0, 1.0], [0.5, 3.0],[0.001, 2.5], Δd) # X plane



# Dislay the labaled array to verify that the array is correctly created for fluid, walls, edges and corners.
#display(Labeled_tlm)


# Begin propagation for a given total time ´T´ which is the first parameter of the function

freq = 100
fs = 6000
Δd =  c / fs
Δt = (Δd / c)
N = Int(Time ÷ Δt)
fft_mic = zeros((N))



#mic_array = zeros(Float64, 31, N)

Labeled_tlm, pressure_grid, SN, SE, SS, SW, SU, SD, IN, IE, IS, IW, IU, ID = create_shoebox(Δd, 4.0, 3.0, 2.5)
# Verify that the given distance value satisfy a given amount of points per wavelength of the given frequency


#boxshape = draw_box_closed([0.5, 2.65, 1.2], 5, d_box, b_box, h_box, N_QRD, b_QRD, d_max_QRD, d_vegger, d_bakplate, d_skilleplate, d_absorbent_QRD)





# Iterate through the box shape and update the labaled array with the new values for the box.

"""
for (i,plane) in enumerate(boxshape)
   
    println("w",i, "\t", plane)
    
    place_wall(Labeled_tlm, plane[1], plane[2], plane[3], Δd)
    
end

"""

#check_fs(fs)
iterate_grid(Time, Δd, pressure_grid, SN, SE, SS, SW, SU, SD, IN, IE, IS, IW, IU, ID)
