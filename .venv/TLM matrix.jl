using PlotlyJS
using Plots
using Printf
using DSP
using Statistics, StatsPlots
using FileIO
using ProgressMeter
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
include("Plot_heatmap.jl")



"""
GMSH - Tool for Julia - Mesh generator
Sketchup - Draw grid/walls


--> Check slice in other dimension 

Time weighted L_fast 
"""


####### GLOBAL PARAMETERS ########
Temp = 291
ρ_air = 1.225
c = 343.2*sqrt(Temp/293)
freq = 300
fs = 3000
po = 20e-6
p0 = 20e-6
#R = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
R = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
# R = [1,1,1,1,1,1,1]


# What type of source should be used. priority directional -> harmonic -> impulse
impulse = true
harmonic = false
harmonic_directional = false

# Positions for the different sources and microphone positions. Given in meters
harm_pos = [2.2,1.8,1.2]
imp_pos = [2.1,1.8,1.2]
meas_position1 = [2.3,2.0,1.2]
meas_position2 = [3.2,1.8,1.2]
meas_position3 = [3.2,2.4,1.2]

# Strength value for the impulse value and harmonic source as well as the total time the source should be on
# Strength given in Pascal
imp_val_p = 1000
signal_strength_Pa = 100
time_source = 0.2




######################################

####### Box parameters ###############
d_box = 0.25
b_box = 0.3
h_box = 1.8
N_QRD = 7
d_max_QRD = 0.1
d_vegger = 0.098
d_bakplate = 0.098
d_skilleplate = 0.060
d_absorbent_QRD = 0.09
b_QRD = (b_box-2*d_vegger) / 7
#######################################


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
    start_index = it
    end_index = length(pressure)
    for i in it:length(pressure)
        if pressure[i] <= pressure[it] - 5
            start_index = i
            break
        end
    end
    for i in start_index:length(pressure)
        if pressure[i] <= pressure[it] - 60
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
    Plots.plot(time, pressure, xlabel="Time (s)", ylabel="Pressure (dB)", label="Pressure",yaxis=:log)
    #Plots.plot!(x, b₀ .+ b₁ .* x, label="Regression Line",yaxis=:log)
    #Plots.hline!([pressure[start_index], pressure[end_index]], linestyle=:dash, label="Regression bounds, T60",yaxis=:log)
    Plots.savefig(string("R=",R[1], " ",freq, " Hz  fs ",fs , " source ", impulse ? "Impulse" : "Harmonic","T60 ",t60,".png"))
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
    #ylims!((0,120))
    Plots.savefig(string("R=",R[1], " ",freq, " Hz  fs ",fs ," source ", impulse ? "Impulse" : "Harmonic",".png"))
end


function _time_to_samples(time::Float64, Δt::Float64)
    return Int(time÷Δt)
end



function iterate_grid(T::Float64, Δd, pressure_grid::Array{Float64,3}, SN::Array{Float64,3}, SE::Array{Float64,3}, SS::Array{Float64,3}, SW::Array{Float64,3}, SU::Array{Float64,3}, SD::Array{Float64,3},PN::Array{Float64,3}, PE::Array{Float64,3}, PS::Array{Float64,3}, PW::Array{Float64,3}, PU::Array{Float64,3}, PD::Array{Float64,3})
    
    # Create time step value, total iteration number N and the sizes of the matrix
    Δt = (Δd / c)
    N = Int(T ÷ Δt)
    L = size(pressure_grid,1)
    M = size(pressure_grid,2)
    N_length = size(pressure_grid,3)

    println("L: ", L,"\tM ", M, "\tN", N_length)
    println("Time interval between each step: ", 1000*Δt, " ms")
    println("The tot iteration number is: ", N)

    # Generate white noise with the sound level, in dB, given by ´sound_level´
    sound_level = 144.0
    WHITE_NOISE = generate_noise_signal(N, Δt, sound_level)

    # Create arrays for data collection for plots and heatmaps.
    p_arr = Float64[]
    meas_pos1 = []
    meas_pos2 = []
    meas_pos3 = []
    gif_arr_x = []
    gif_arr_y = []
    gif_arr_z = []

    # Create coordinates for the microphone position. Turn real coordinates into indicies
    ind1 = _real_to_index(meas_position1, Δd)
    ind2 = _real_to_index(meas_position2, Δd)
    ind3 = _real_to_index(meas_position3, Δd)

    # Track the progress using progressmeter to better visualize ETA
    prog1 = Progress(N)


    ######## Step 1 - Insert energy into grid

    # Iterate through the matrix with timesteps "n" for a given total time "T"
    for n in 1:N
        #println("Step: ", n)
        # Step 1 - Insert energy into grid
        
        # Use the global boolean parameter ´impulse´ if the initial scattering matrix will have an impulse signal
        if impulse == true && n == 1
            println("Inserting Impulse")
            i = _real_to_index(imp_pos,Δd)[1]
            j = _real_to_index(imp_pos,Δd)[2]
            k = _real_to_index(imp_pos,Δd)[3]
            PN[i,j,k] = imp_val_p
            PE[i,j,k] = imp_val_p
            PS[i,j,k] = imp_val_p
            PW[i,j,k] = imp_val_p
            PU[i,j,k] = imp_val_p
            PD[i,j,k] = imp_val_p
        end

         # Use the global boolean parameter ´harmonic´ if the grid should experience a harmonic time signal - Will work as a point source located at a single node 
        if (harmonic == true) && (n <= _time_to_samples(time_source,Δt))

            if n == 1
                println("Insert Harmonic signal")
                println("Time_to_samples for",time_source, "s - ", _time_to_samples(time_source,Δt))
            end
            # Locate the discrete indicies from the real positions
            i = _real_to_index(harm_pos,Δd)[1]
            j = _real_to_index(harm_pos,Δd)[2]
            k = _real_to_index(harm_pos,Δd)[3]
            PN[i,j,k] = pressure_value(n*Δt)
            PE[i,j,k] = pressure_value(n*Δt)
            PS[i,j,k] = pressure_value(n*Δt)
            PW[i,j,k] = pressure_value(n*Δt)
            PU[i,j,k] = pressure_value(n*Δt)
            PD[i,j,k] = pressure_value(n*Δt)
        end

        # Use the global boolean parameter harmonic_directional if the grid should experience a harmoncic time signal in only one direction / Weaker in some directions - Will be more general in the future.
        if harmonic_directional == true
            # Locate the discrete indicies from the real positions
            i = harm_pos[1]
            j = harm_pos[2]
            k = harm_pos[3]
            PN[i,j,k] = pressure_value_directional(n*Δt)[1]
            PE[i,j,k] = pressure_value_directional(n*Δt)[2]
            PS[i,j,k] = pressure_value_directional(n*Δt)[3]
            PW[i,j,k] = pressure_value_directional(n*Δt)[4]
            PU[i,j,k] = pressure_value_directional(n*Δt)[5]
            PD[i,j,k] = pressure_value_directional(n*Δt)[6]
        end
        
        ######## Step 2 - Calculate overall pressure
        pressure_grid = overal_pressure(PN, PE, PS, PW, PU, PD)


        ######## Step 2.1 - Insert values into different arrays for plots and animation
        meas1 = pressure_grid[ind1[1],ind1[2],ind1[3]]
        meas2 = pressure_grid[ind2[1],ind2[2],ind2[3]]
        meas3 = pressure_grid[ind3[1],ind3[2],ind3[3]]

        push!(gif_arr_x, 10*log10.(pressure_grid[ind1[1],:,:].^2))
        push!(gif_arr_y, 10*log10.(pressure_grid[:,ind1[2],:].^2))
        push!(gif_arr_z, 10*log10.(pressure_grid[:,:,ind1[3]].^2))

        push!(meas_pos1, meas1)
        push!(meas_pos2, meas2)
        push!(meas_pos3, meas3)

        push!(p_arr, sqrt((1/(L*M*N_length))*sum(pressure_grid.^2)))





        ######## Step 3 - Calculate the scattering matrix
        SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD = calculate_scattering_matrix(SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)



        ######## Step 4 - Calculate the pressure matrix
        SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD = calculate_pressure_matrix(Labeled_tlm, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)
        
       


        ####### Checking for places where energy is NOT conserved
        Conservation_checker_array = (PN .- SN) .+ (PE .- SE) .+ (PS .- SS) .+ (PE .- SE) .+ (PU .- SU) .+ (PD .- SD) 
        positions = findall(x -> x != 0, Conservation_checker_array)
        
        println("Harmonic source position: ", _real_to_index(harm_pos,Δd))
        for pos in positions
            println(pos)
        end
        





        # next iteraton for the progressmeter bar
        
        next!(prog1)
        readline()
    end

    # Indicate that propagation is done
    println("Finished with propagation")


    # Create a time vector ´x´ for the total discrete time ´N´
    x = collect(range(0, stop=(N-1)*Δt, step=Δt))


    
   
    # Create the time weigthed sound power level using the ´ltau´ function from Guillaume D.
    p1 = ltau(meas_pos1, fs, 10*Δt, p0)
    p1_2 = ltau(meas_pos2, fs, 2*Δt, p0)
    p1_3 = ltau(meas_pos3, fs, 2*Δt, p0)

    # Create a pressure array using the ´_Lp´ function
    p_arr = _Lp(p_arr)
    Plots.plot(x, 20*log10.(sqrt.(meas_pos2.^2)), xlabel="Time (s)", ylabel="Pressure (dB)", label="Pressure")
    # Different actions to be done on the finished result. remve comment mark and edit the input parameters to alter the functions.
    #find_t60(meas_pos2.^2, x, _time_to_samples(time_source,Δt)-10, L*Δd, M*Δd, N_length*Δd)
    #plot_arrays(p_arr, p_arr, p_arr, Δt)
    #_heatmap_gif11(gif_arr_x, N,x, Δd, string("R=",R[1], " ",freq, " Hz  fs ",fs , " ",T," s source ", impulse ? "Impulse" : "Harmonic" ," x plane.gif"))
    #_heatmap_gif11(gif_arr_y, N,x, Δd, string("R=",R[1], " ",freq, " Hz  fs ",fs , " ",T," s source ", impulse ? "Impulse" : "Harmonic" ," y plane.gif"))
    #_heatmap_gif11(gif_arr_z, N,x, Δd, string("R=",R[1], " ",freq, " Hz  fs ",fs , " ",T," s source ", impulse ? "Impulse" : "Harmonic" ," z plane.gif"))

end





###### Main code 

# create the distance value between each node in the TLM matrix
Δd =  c / fs





# Verify that the given distance value satisfy a given amount of points per wavelength of the given frequency
check_fs(fs)




# Create the pressure- and scattering arrays as well as the labaled arrays and the pressure grid.
Labeled_tlm, pressure_grid, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD = create_shoebox(Δd, 4.0, 3.0, 2.5)




# Create the shape of the box given the start positions and the global box parameters given in the top of the file - The box
# is created by starting from the bottom left of the box and goes clockwise depending on the position value used, that define the surface to place the box to.
boxshape = draw_box_closed([0.2, 1.5, 0.2], 1, d_box, b_box, h_box, N_QRD, b_QRD, d_max_QRD, d_vegger, d_bakplate, d_skilleplate, d_absorbent_QRD)


# Iterate through the box shape and update the labaled array with the new values for the box.

"""
for (i,plane) in enumerate(boxshape)
   
    println("w",i, "\t", plane)
    
    place_wall(Labeled_tlm, plane[1], plane[2], plane[3], Δd)
    
end
"""

# Dislay the labaled array to verify that the array is correctly created for fluid, walls, edges and corners.
#display(Labeled_tlm)


# Begin propagation for a given total time ´T´ which is the first parameter of the function
iterate_grid(0.9, Δd, pressure_grid, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)


