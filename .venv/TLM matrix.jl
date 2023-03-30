using LinearAlgebra
using Plots



"""
GMSH - Tool for Julia - Mesh generator
Sketchup - Draw grid/walls


Time weighted L_fast 
"""


####### GLOBAL PARAMETERS ########
Temp = 291
ρ_air = 1.225
c = 343.2*sqrt(Temp/293)

impulse = false
imp_pos = [20,20,20]
imp_val_p = 10
harmonic = true
harm_pos = [20,20,20]
harmonic_directional = false
freq = 100
po = 2*10^(-5)
A = 20
fs = 3000

R = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
##################################


function TLM(Δd, height, width, length)
    nx = length ÷ Δd
    ny = width ÷ Δd
    nz = height ÷ Δd

    tlm = zeros((nx,ny,nz))
    return tlm
end



# Functioning code for creating a shoebox shaped room having labels for all nodes describing if its a fluid, 
# surface, edge or corner. 
function create_shoebox(Δd, height, width, length)
    
    nx = Int(length ÷ Δd + 1)
    ny = Int(width ÷ Δd +1)
    nz = Int(height ÷ Δd +1)

    println("Creating Shoebox shape")
    println("size:\t x_direction: ", nx*Δd, " m \t y_direction: ",ny*Δd," m \t z_direction: ", nz*Δd, " m")

    box = zeros(Int, (nx,ny,nz))

    A = []

    B = A
    C = A
    D = A

    
    SN = zeros((nx,ny,nz))
    SE = zeros((nx,ny,nz))
    SS = zeros((nx,ny,nz))
    SW = zeros((nx,ny,nz))
    SU = zeros((nx,ny,nz))
    SD = zeros((nx,ny,nz))
    PN = zeros((nx,ny,nz))
    PE = zeros((nx,ny,nz))
    PS = zeros((nx,ny,nz))
    PW = zeros((nx,ny,nz))
    PU = zeros((nx,ny,nz))
    PD = zeros((nx,ny,nz))


    pressure_grid = zeros((nx,ny,nz))


    # Set surface values by checking the position
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz

                if i == 1
                    box[i, j, k] = 1
                elseif i == nx
                    box[i, j, k] = 2
                elseif j == 1
                    box[i, j, k] = 3
                elseif j == ny
                    box[i, j, k] = 4
                elseif k == 1
                    box[i, j, k] = 5
                elseif k == nz
                    box[i, j, k] = 6
                end
            end
        end
    end


    # Set Edge values
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz

                if i == 1 && j == ny
                    box[i, j, k] = 14
                elseif j == ny && k == 1
                    box[i, j, k] = 45
                elseif i == nx && j == ny
                    box[i, j, k] = 24
                elseif j == ny && k == nz
                    box[i, j, k] = 46
                elseif i == nx && k == 1
                    box[i, j, k] = 25
                elseif i == nx && k == nz
                    box[i, j, k] = 26
                elseif i == nx && j == 1
                    box[i, j, k] = 23
                elseif j == 1 && k == 1
                    box[i, j, k] = 35
                elseif i == 1 && k == 1
                    box[i, j, k] = 15
                elseif i == 1 && k == nz
                    box[i, j, k] = 16
                elseif i == 1 && j == 1
                    box[i, j, k] = 13
                elseif j == 1 && k == nz
                    box[i, j, k] = 36
                end
            end
        end
    end

    # Set Corner values
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz

                if i == nx && j == 1 && k == 1
                    box[i, j, k] = 235
                elseif i == 1 && j == 1 && k == 1
                    box[i, j, k] = 135
                elseif i == 1 && j == ny && k == 1
                    box[i, j, k] = 145
                elseif i == nx && j == ny && k == 1
                    box[i, j, k] = 245
                elseif i == nx && j == 1 && k == nz
                    box[i, j, k] = 236
                elseif i == 1 && j == 1 && k == nz
                    box[i, j, k] = 136
                elseif i == 1 && j == ny && k == nz
                    box[i, j, k] = 146
                elseif i == nx && j == ny && k == nz
                    box[i, j, k] = 246
                end
            end
        end
    end
    return box, pressure_grid, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD
end


function case(n::Int64)
    Diffusor = false
    if n < 0
        Diffusor = true
        n = abs(n)
    end
    num_str = string(n)
    used_nums = [parse(Int, digit) for digit in num_str]
    all_nums = Set(1:6)
    unused_nums = setdiff(collect(all_nums), used_nums)

    return used_nums, unused_nums, Diffusor
end


"""
Old function to handle case parsing. --> Did only consider used labels, changed with function above that consider both used and unused

function case(n::Integer)
    Diffusor = false
    if n < 0
        Diffusor = true
        n = abs(n)
    end
    digits_set = Set(Int[])
    while n > 0
        digit = n % 10
        push!(digits_set, digit)
        n = div(n, 10)    
    end


    return digits_set, Diffusor
end
"""

function mark_box_walls(arr::Array{Float64,3}, d, corners)
    for i in 1:size(arr, 1)
        for j in 1:size(arr, 2)
            for k in 1:size(arr, 3)
                node_pos = [(i-1)*d, (j-1)*d, (k-1)*d]
                for n in 1:6
                    corner1 = corners[n,:]
                    corner2 = corners[n+1,:]
                    dist1 = LinearAlgebra.norm(corner1 - node_pos)
                    dist2 = LinearAlgebra.norm(corner2 - node_pos)
                    surface_dist = abs(dist1 - dist2)
                    if surface_dist < d/10
                        arr[i,j,k] = n
                        break
                    end
                end
            end
        end
    end
    return arr
end

function merges(A::Int64, B::Int64)::Int64
    B_digits = [parse(Int64, digit) for digit in string(B)]
    A_digits = [parse(Int64, digit) for digit in string(A)]
    for digit in B_digits
        if !(digit in A_digits)
            push!(A_digits, digit)
        end
    end
    return parse(Int64, join(sort(A_digits)))
end


function merge(A::Float64, B::Float64)
    # Convert A and B to integers
    int_A = round(Int, A)
    int_B = round(Int, B)

    # Increment int_A by int_B
    int_result = int_A + int_B

    # Convert int_result to string to remove duplicates
    str_result = string(int_result)

    # Remove duplicates and convert back to integer
    int_result = parse(Int, join(Set(str_result)))

    return int_result
end




function place_wall(arr::Array{Float64, 3}, x_pos::Array{Float64, 1}, y_pos::Array{Float64, 1}, z_pos::Array{Float64, 1}, Δd::Float64)
    
    

    if x_pos[1]==x_pos[2]
        println("X pos is plane")
        for i in 1:size(arr,1)
            for j in Int(ceil(y_pos[1]/Δd)):Int(floor(y_pos[2]/Δd))
                for k in Int(ceil(z_pos[1]/Δd)):Int(floor(z_pos[2]/Δd))
                    B = arr[i,j,k]
                    if (((i*Δd - x_pos[1] >= -Δd)) && ((i*Δd - x_pos[1]) < 0))
                        X = 2.0
                        arr[i,j,k] = merge(B,X)
                    elseif (((i*Δd - x_pos[1]) <= Δd) && ((i*Δd - x_pos[1]) > 0))
                        X = 1.0
                        arr[i,j,k] = merge(B,X)
                        
                    end
                        
                end
            end
        end

        
    elseif y_pos[1] == y_pos[2]
        println("Y pos is plane")
        for i in Int(ceil(x_pos[1]/Δd)):Int(floor(x_pos[2]/Δd))
            for j in 1:size(arr,2)
                for k in Int(ceil(z_pos[1]/Δd)):Int(floor(z_pos[2]/Δd))
                    B = arr[i,j,k]
                    if (((j*Δd - y_pos[1]) >= -Δd) && ((j*Δd - y_pos[1]) < 0))
                        Y = 4.0
                        arr[i,j,k] = merge(B,Y)
                    elseif (((j*Δd - y_pos[1]) <= Δd) && ((j*Δd - y_pos[1]) > 0))
                        Y = 3.0
                        arr[i,j,k] = merge(B,Y)
                    end
                        
                end
            end
        end
    elseif z_pos[1] == z_pos[2]
        println("Z pos is plane")
        for i in Int(ceil(x_pos[1]/Δd)):Int(floor(x_pos[2]/Δd))
            for j in Int(ceil(y_pos[1]/Δd)):Int(floor(y_pos[2]/Δd))
                for k in 1:size(arr,3)
                    B = arr[i,j,k]
                    if (((k*Δd - z_pos[1]) >= -Δd) && ((k*Δd - z_pos[1]) < 0))
                        Z = 6.0
                        arr[i,j,k] = merge(B,Z)
                    elseif (((k*Δd - z_pos[1]) <= Δd) && ((k*Δd - z_pos[1]) > 0))
                        Z = 5.0
                        arr[i,j,k] = merge(B,Z)

                    end
                        
                end
            end
        end
    else
        println("Not a wall")
    end

    return arr
end

function calculate_pressure_matrix(Labeled_tlm::Array{Int64,3}, SN::Array{Float64,3}, SE::Array{Float64,3}, SS::Array{Float64,3}, SW::Array{Float64,3}, SU::Array{Float64,3}, SD::Array{Float64,3},PN::Array{Float64,3}, PE::Array{Float64,3}, PS::Array{Float64,3}, PW::Array{Float64,3}, PU::Array{Float64,3}, PD::Array{Float64,3})
    for i in 1:size(Labeled_tlm,1)
        for j in 1:size(Labeled_tlm,2)
            for k in 1:size(Labeled_tlm,3)
                if Labeled_tlm[i,j,k] == 0
                    #println("Label = 0")
                    PN[i,j,k] = SS[i - 1, j, k]
                    PE[i,j,k] = SW[i, j + 1, k]
                    PS[i,j,k] = SN[i + 1, j, k]
                    PW[i,j,k] = SE[i, j - 1, k]
                    PU[i,j,k] = SD[i, j, k + 1]
                    PD[i,j,k] = SU[i, j, k - 1]
                else
                    case_used, case_unused, diffusor = case(Labeled_tlm[i,j,k])
                    #println("Case used: ", case_used, " - Case unused: ", case_unused, " - Diffusor: ", diffusor)
                    
                    for n in case_used
                        #println("Case - used: ",n)
                        Refl = diffusor ? R[6] : R[n]
                        #println("Reflection value: ", Refl)
                        if n == 1
                            PN[i,j,k] = Refl * SN[i,j,k]                        
                        elseif n == 2
                            PS[i,j,k] = Refl * SS[i,j,k]
                        elseif n == 3
                            PW[i,j,k] = Refl * SW[i,j,k]
                        elseif n == 4
                            PE[i,j,k] = Refl * SE[i,j,k]
                        elseif n == 5
                            PD[i,j,k] = Refl * SD[i,j,k]
                        elseif n == 6
                            PU[i,j,k] = Refl * SU[i,j,k]
                        end
                    end
                    for n in case_unused
                        #println("Case - unused: ", n)
                        if n == 1
                            PN[i,j,k] = SS[i - 1, j, k]
                        elseif n == 2
                            PS[i,j,k] = SN[i + 1, j, k]
                        elseif n == 3
                            PW[i,j,k] = SE[i, j - 1, k]
                        elseif n == 4
                            PE[i,j,k] = SW[i, j + 1, k]
                        elseif n == 5
                            PD[i,j,k] = SU[i, j, k - 1]
                        elseif n == 6
                            PU[i,j,k] = SD[i, j, k + 1]
                        end
                    end
                end
            end
        end
    end
    
    return SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD
end


function calculate_scattering_matrix(SN::Array{Float64,3}, SE::Array{Float64,3}, SS::Array{Float64,3}, SW::Array{Float64,3}, SU::Array{Float64,3}, SD::Array{Float64,3},PN::Array{Float64,3}, PE::Array{Float64,3}, PS::Array{Float64,3}, PW::Array{Float64,3}, PU::Array{Float64,3}, PD::Array{Float64,3})
    SW = (1/3) * ((-2)*PW .+ PN .+ PE .+ PS .+ PU .+ PD)
    SN = (1/3) * (PW .- (2*PN) .+ PE .+ PS .+ PU .+ PD)
    SE = (1/3) * (PW .+ PN .- (2*PE) .+ PS .+ PU .+ PD)
    SS = (1/3) * (PW .+ PN .+ PE .- (2*PS) .+ PU .+ PD)
    SD = (1/3) * (PW .+ PN .+ PE .+ PS .- (2*PU) .+ PD)
    SU = (1/3) * (PW .+ PN .+ PE .+ PS .+ PU .- (2*PD))
    return SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD
end


"""
# First try at replacing surfaces with the value 1 ---- trashed due to unecessary complexity

function replace_edges_with_one(arr::AbstractArray{T,N}) where {T,N}
    for i in 1:N
        arr = cat(1, arr[1], fill(one(T), (1, size(arr, i) - 2)), arr[end])
        arr = cat(2, arr[:, 1], fill(one(T), (size(arr, 1) - 2, 1)), arr[:, end])
        if ndims(arr) > 2
            arr = cat(3, arr[:, :, 1], fill(one(T), (size(arr, 1), size(arr, 2), 1)), arr[:, :, end])
        end
    end
    return arr
end

"""

function pressure_value(t::Float64)
    return round(A*sin(t*2*pi*freq);digits=4)
end

function pressure_value_directional(t::Float64)
    return A*sin(t*2*pi*freq)*directivity_factor
end


function overal_pressure(PN::Array{Float64,3}, PE::Array{Float64,3}, PS::Array{Float64,3}, PW::Array{Float64,3}, PU::Array{Float64,3}, PD::Array{Float64,3})
    P_grid = (1/3) * (PW .+ PN .+ PE .+ PS .+ PU .+ PD)
    return P_grid
end


function time_to_discrete(timeval::Float64, fs::Int64)
    return Int(timeval * fs)
end

function create_splitted(arr::Array{Float64,1}, T::Float64, fs::Int64)
    N_t = time_to_discrete(T,fs)
    Length = size(arr,1)
    N = Int(floor(Length / N_t))
    L_eq_arr = zeros((N,N_t))

    # Split the 1D array into a 2D array that is cut off at the end to be a whole integer step from N_t
    for i in 1:N
      for j in 1:N_t
        L_eq_arr[i,j] = arr[(i-1)*N_t + j]
      end
    end
    return L_eq_arr
  end

  """
function create_arrays(data::Array{Float64,1}, fs::Float64, T::Float64)

    # Determine the length of the output arrays
    len = floor(Int, T * length(data))

    # Pad the input array with zeros
    array1 = vcat(data, zeros(T * length(data) - length(data)))

    # Remove some of the last values from the input array
    array2 = data[1:(len-1)]

    return (array1, array2)
end
"""

function linspace(start::Number, stop::Number, n::Integer)
    step = (stop - start) / (n - 1)
    return range(start, stop=stop, length=n)
end

function Leq_fast(arr::Array{Float64,1}, fs::Int64)
    N = time_to_discrete(0.015, fs)
    arr1 = create_splitted(arr, 0.015, fs)
    it = size(arr,1) ÷ N
    l_eq = []
    println(size(arr1))


    # Creates the new L_eq array that contains pressure data for the given timefactor : fast = 0.125 s
    for i in 1:size(arr1,1)
        sum = 0
        for j in 1:size(arr1,2)
            sum += (arr1[i,j] / po)^2
        end
        push!(l_eq, 10*log(10,sum / size(arr1,2))) 

    end
    # Create the timestep values to be used when plotting the L_eq values with real time scale.
    time = linspace(0, size(arr,1) / fs, it)

    println("Calculated Leq")

    return l_eq, time
end






function get_t30_db_plot(A,time)
    # Convert decibel values to pressure values
    p = 10 .^ (A / 20)

    # Calculate the power of the pressure data
    power = p .^ 2

    # Find the index where the power starts to drop
    peak_index = argmax(power)
    drop_index = findfirst(x -> x < power[peak_index]/2, power[peak_index:end])
    if isempty(drop_index)
        error("Could not detect drop in sound power.")
    end
    t0 = drop_index + peak_index - 3

    # Find the indices where the pressure has dropped by 5 dB and 35 dB
    p5_index = findfirst(x -> x < A[t0] - 5, A[t0:end])
    p35_index = findfirst(x -> x < A[t0] - 35, A[t0:end])
    if isempty(p5_index) || isempty(p35_index)
        error("Could not find points where pressure has dropped by 5 dB or 35 dB.")
    end
    t5 = p5_index + t0 - 1
    t35 = p35_index + t0 - 1

    println("t5: ", t5)
    println("t35: ", t35)

    # Calculate the slope and intercept of the regression line
    x = time[t5:t35]
    y = A[t5:t35]
    n = length(x)
    sx = sum(x)
    sy = sum(y)
    sxy = sum(x .* y)
    sx2 = sum(x .^ 2)
    m = (n * sxy - sx * sy) / (n * sx2 - sx^2)
    b = (sy - m * sx) / n

    println("m: ",m,"\t b: ",b)

    # Calculate the decay rate of the sound over time
    # from the slope of the regression line.
    decay_rate = -20 * m

    # Calculate T30 from the decay rate
    #t30 = 30 / decay_rate
    t30 = time[t35] - time[t5]
    
    println("RT: ",t30*2)

    # Plot the pressure values and the regression line
    p1 = plot(time, A, label="Pressure (dB)", grid=true, title="Pressure plot", xlabel="Time [s]", ylabel="Pressure [dB]")
    p2 = plot!(x, m*x .+ b, label="Regression line, T30")
    p3 = hline!([A[t5], A[t35]], linestyle=:dash, label="Regression bounds, T30")
    
    return t30, p1, p2, p3
end



function time_weighted_sound_level_t(pressure_data::Array{Float64}, reference_pressure::Float64)
    # Set up constants according to IEC 61672-1
    t1 = 0.125  # Time constant for fast weighting (s)
    t2 = 1.0    # Time constant for slow weighting (s)
    k = 1.0     # Factor for combining fast and slow weightings

    # Initialize variables
    Lp_fast = zeros(length(pressure_data))
    Lp_slow = zeros(length(pressure_data))
    Lp_time_weighted = zeros(length(pressure_data))

    # Calculate sound pressure level for each sample using fast weighting
    for i in 1:length(pressure_data)
        Lp_fast[i] = 20 * log10(abs(pressure_data[i]) / reference_pressure) + 2.0
    end

    # Calculate sound pressure level for each sample using slow weighting
    for i in 1:length(pressure_data)
        if i == 1
            Lp_slow[i] = Lp_fast[i]
        else
            Lp_slow[i] = k * Lp_slow[i-1] + (1-k) * Lp_fast[i]
        end
    end

    # Calculate time-weighted sound level using the calculated sound pressure levels
    for i in 1:length(pressure_data)
        Lp_time_weighted[i] = 20 * log10(sqrt(sum(pressure_data[1:i].^2)/i) / reference_pressure)
    end

    return Lp_time_weighted
end






function iterate_grid(T::Float64, Δd, pressure_grid::Array{Float64,3}, SN::Array{Float64,3}, SE::Array{Float64,3}, SS::Array{Float64,3}, SW::Array{Float64,3}, SU::Array{Float64,3}, SD::Array{Float64,3},PN::Array{Float64,3}, PE::Array{Float64,3}, PS::Array{Float64,3}, PW::Array{Float64,3}, PU::Array{Float64,3}, PD::Array{Float64,3})
    Δt = (Δd / c)
    println("Sampling frequency fs: ", 1 / Δt)
    N = T ÷ Δt
    println("The tot iteration number is: ",N)
    p_node = [10, 10, 10]
    p_arr = Float64[]

    # Step 1 - Insert energy into grid
    
    # Iterate through the matrix with timesteps "n" for a given total time "T"
    for n in 1:N
        # Step 1 - Insert energy into grid
        
        # Use the global boolean parameter impulse if the initial scattering matrix will have an impulse signal
        if impulse == true && n == 1
            println("Inserting Impulse")
            i = imp_pos[1]
            j = imp_pos[2]
            k = imp_pos[3]
            PN[i,j,k] = imp_val_p
            #PE[i,j,k] = imp_val_p*10000
            #PS[i,j,k] = imp_val_p*1000
            #PW[i,j,k] = imp_val_p*1000
            #PU[i,j,k] = imp_val_p*10000
            #PD[i,j,k] = imp_val_p*1
        end

         # Use the global boolean parameter harmonic if the grid should experience a harmonic time signal - Will work as a point source located at a single node 
        if harmonic == true && n < 1200
            if n == 1
                println("Insert Harmonic signal")
            end
            i = harm_pos[1]
            j = harm_pos[2]
            k = harm_pos[3]
            PN[i+1,j,k] = pressure_value(n*Δt)
            PE[i,j-1,k] = pressure_value(n*Δt)
            PS[i-1,j,k] = pressure_value(n*Δt)
            PW[i,j+1,k] = pressure_value(n*Δt)
            PU[i,j,k-1] = pressure_value(n*Δt)
            PD[i,j,k+1] = pressure_value(n*Δt)
        end

        # Use the global boolean parameter harmonic_directional if the grid should experience a harmoncic time signal in only one direction / Weaker in some directions - Will be more general in the future.
        if harmonic_directional == true
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
        
        # Step 2 - Calculate overall pressure
        pressure_grid = overal_pressure(PN, PE, PS, PW, PU, PD)
        println("Step:\t",n)
       #println(pressure_grid)
        push!(p_arr, pressure_grid[15,14,14])

        """
        println("PN : ",PN)
        println("PE : ",PE)
        println("PS : ",PS)
        println("PW : ",PW)
        println("PU : ",PU)
        println("PD : ",PD)
        println("************************************")
        println("SN : ",SN)
        println("SE : ",SE)
        println("SS : ",SS)
        println("SW : ",SW)
        println("SU : ",SU)
        println("SD : ",SD)

        #println("SE : ",SE)

        #println("P_grid : ", pressure_grid)
        """

        # Step 3 - Calculate the scattering matrix
        SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD = calculate_scattering_matrix(SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)

        # Step 4 - Calculate the pressure matrix
        SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD = calculate_pressure_matrix(Labeled_tlm, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)
    end

    x = range(0, stop=(N-1)*Δt, step=Δt)
    #println("Pressure array: ",p_arr)
    println("Finished with propagation")

    println("Calculating L_eq")
    t1 = time_weighted_sound_level_t(p_arr, p0)


    #l_eq, time = Leq_fast(p_arr,fs)

    #println(p_arr)
    println(x)
    
    #t30, p1, p2, p3 = get_t30_db_plot(l_eq,  time)

    display(plot(x, t1))
end



Δd = c / fs

Labeled_tlm, pressure_grid, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD = create_shoebox(Δd, 2.5, 3, 4.0)

iterate_grid(1.2, Δd, pressure_grid, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)