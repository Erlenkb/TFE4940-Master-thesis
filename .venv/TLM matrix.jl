
R = [1, 1, 1, 1, 1, 1, 1]

"""
function TLM(Δd, height, width, length)
    nx = length ÷ Δd
    ny = width ÷ Δd
    nz = height ÷ Δd

    tlm = zeros((nx,ny,nz))
end

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
    nx = length ÷ Δd
    ny = width ÷ Δd
    nz = height ÷ Δd
    box = zeros(Int, nx, ny, nz)

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

    
    return box
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
IW = TLM(1, 4, 3, 3);

IWE = create_shoebox(1,7,7,7)

