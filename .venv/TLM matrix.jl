using LinearAlgebra
using Plots


"""
GNSH - Tool for Julia - Mesh generator
Sketchup - Draw grid/walls
"""


####### GLOBAL PARAMETERS ########

Temp = 291
ρ_air = 1.225
c = 343.2*sqrt(Temp/293)

impulse = true
imp_pos = [1,1,1]
imp_val_p = 1
harmonic = false
harm_pos = [4,4,4]
harmonic_directional = false
freq = 500
po = 2*10^(-5)
A = 1

R = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


##################################


function TLM(Δd, height, width, length)
    nx = length ÷ Δd
    ny = width ÷ Δd
    nz = height ÷ Δd

    tlm = zeros((nx,ny,nz))
    return tlm
end
"""
function Create_Room(Δd, height, width, length)
    nx = length ÷ Δd
    ny = width ÷ Δd
    nz = height ÷ Δd
    box = zeros(Int, nx, ny, nz)
    box[[1,end],:,:] .= 1
    box[:,[1,end],:] .= 1
    box[:,:, [1,end]] .= 1
    return box
end
"""



# Functioning code for creating a shoebox shaped room having labels for all nodes describing if its a fluid, 
# surface, edge or corner. 
function create_shoebox(Δd, height, width, length)
    
    nx = Int(length ÷ Δd - 1)
    ny = Int(width ÷ Δd - 1)
    nz = Int(height ÷ Δd - 1)

    box = zeros(Int, (nx,ny,nz))
    box1 = zeros((nx,ny,nz))

    SN = box1
    SE = box1
    SS = box1
    SW = box1
    SU = box1
    SD = box1
    PN = box1
    PE = box1
    PS = box1
    PW = box1
    PU = box1
    PD = box1
    pressure_grid = box1


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
                    PN[i,j,k] = SS[i - 1, j, k]
                    PE[i,j,k] = SW[i, j + 1, k]
                    PS[i,j,k] = SN[i + 1, j, k]
                    PW[i,j,k] = SE[i, j - 1, k]
                    PU[i,j,k] = SD[i, j, k + 1]
                    PD[i,j,k] = SU[i, j, k - 1]
                    continue
                else
                    case_used, case_unused, diffusor = case(Labeled_tlm[i,j,k])
                    
                    for n in case_used
                        Refl = diffusor ? R[6] : R[n]
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
    SW = (1/3) * (-2*PW .+ PN .+ PE .+ PS .+ PU .+ PD)
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
    return A*sin(t*2*pi*freq)
end

function pressure_value_directional(t::Float64)
    return A*sin(t*2*pi*freq)*directivity_factor
end


function overal_pressure(PN::Array{Float64,3}, PE::Array{Float64,3}, PS::Array{Float64,3}, PW::Array{Float64,3}, PU::Array{Float64,3}, PD::Array{Float64,3})
    P_grid = (1/3) * (PW .+ PN .+ PE .+ PS .+ PU .+ PD)
    return P_grid
end


function iterate_grid(T::Float64, Δd, pressure_grid::Array{Float64,3}, SN::Array{Float64,3}, SE::Array{Float64,3}, SS::Array{Float64,3}, SW::Array{Float64,3}, SU::Array{Float64,3}, SD::Array{Float64,3},PN::Array{Float64,3}, PE::Array{Float64,3}, PS::Array{Float64,3}, PW::Array{Float64,3}, PU::Array{Float64,3}, PD::Array{Float64,3})
    Δt = (Δd / c)
    N = T ÷ Δt
    p_node = [10, 10, 10]
    p_arr = zeros(Int(N))

    # Step 1 - Insert energy into grid
        
    # Use the global boolean parameter impulse if the initial scattering matrix will have an impulse signal
    if impulse
        i = imp_pos[1]
        j = imp_pos[2]
        k = imp_pos[3]
        SN[i,j,k] = imp_val_p
        SE[i,j,k] = imp_val_p
        SS[i,j,k] = imp_val_p
        SW[i,j,k] = imp_val_p
        SU[i,j,k] = imp_val_p
        SD[i,j,k] = imp_val_p
    end

    # Use the global boolean parameter harmonic if the grid should experience a harmonic time signal - Will work as a point source located at a single node 
    if harmonic
        i = harm_pos[1]
        j = harm_pos[2]
        k = harm_pos[3]
        SN[i,j,k] = pressure_value(n*Δt)
        SE[i,j,k] = pressure_value(n*Δt)
        SS[i,j,k] = pressure_value(n*Δt)
        SW[i,j,k] = pressure_value(n*Δt)
        SU[i,j,k] = pressure_value(n*Δt)
        SD[i,j,k] = pressure_value(n*Δt)
    end

    # Use the global boolean parameter harmonic_directional if the grid should experience a harmoncic time signal in only one direction / Weaker in some directions - Will be more general in the future.
    if harmonic_directional
        i = harm_pos[1]
        j = harm_pos[2]
        k = harm_pos[3]
        SN[i,j,k] = pressure_value_directional(n*Δt)[1]
        SE[i,j,k] = pressure_value_directional(n*Δt)[2]
        SS[i,j,k] = pressure_value_directional(n*Δt)[3]
        SW[i,j,k] = pressure_value_directional(n*Δt)[4]
        SU[i,j,k] = pressure_value_directional(n*Δt)[5]
        SD[i,j,k] = pressure_value_directional(n*Δt)[6]
    end

    for n in 1:N
        # Iterate through the matrix with timesteps "n" for a given total time "T"
        
        # Step 2 - Calculate overall pressure
        pressure_grid = overal_pressure(PN, PE, PS, PW, PU, PD)
        p_arr[Int(n)] = pressure_grid[p_node[1],p_node[2],p_node[3]]



        # Step 3 - Calculate the scattering matrix
        SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD = calculate_scattering_matrix(SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)

        # Step 4 - Calculate the pressure matrix
        SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD = calculate_pressure_matrix(Labeled_tlm, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)

    end

    x = range(0, stop=(N-1)*Δt, step=Δt)

    plot(x, p_arr, label="Pressure values at node k")


end




#Labeled_tlm = TLM(1, 10, 10, 10)

#IW = TLM(1, 4, 3, 3)

x = [5.0,5.0]
y = [3.4,6.2]
z = [2.0,4.0]

x2 = [0.65,0.94]
y2 = [1.01,1.01]
z2 = [0.19,0.32]

Δd = 0.1


Labeled_tlm, pressure_grid, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD = create_shoebox(Δd, 5, 3, 3)

#println(Labeled_tlm)


iterate_grid(0.03, Δd, pressure_grid, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)




#Labeled_tlm = place_wall(IWE, x2, y2, z2, Δd)
#IWE3 = place_wall(IWE2, x2, y2, z2, Δd)




