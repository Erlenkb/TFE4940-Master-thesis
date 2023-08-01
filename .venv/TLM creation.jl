
"""
    create_shoebox(Δd, length, width, height)

Takes in a step value between each TLM node, Δd, and a length, width
and height parameter in meters.
Returns a labeled TLM matrix as well as 12 3D arrays for both the scattering matrix
as well as the pressure matrix.
"""




function create_shoebox(Δd, length, width, height)
    
    # Set the discrete lengths of the room 
    nx = Int(length ÷ Δd + 1)
    ny = Int(width ÷ Δd + 1)
    nz = Int(height ÷ Δd + 1)


    println("Creating Shoebox shape")
    println("Δd: ",Δd, " m")
    println("size:\t x_direction: ", nx*Δd, " m \t y_direction: ",ny*Δd," m \t z_direction: ", nz*Δd, " m")

    # Initialize the arrays used for TLM calculations
    box = zeros(Int, (nx,ny,nz))
    pressure_grid = zeros((nx,ny,nz))
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