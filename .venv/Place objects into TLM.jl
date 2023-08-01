
"""
    place_wall(arr, x_pos, y_pos, z_pos, Δd)

Alter the 'arr' array by inserting a plane given by the x_pos, y_pos and z_pos arrays,
containing start-stop values in real coordinates [meters]. It returns the altered array
with elements next to a wall updated with the new label value.

IMPORTANT: The planes is defined in real coordinates, but for all three dimensions, its IMPORTANT
that the lowest value is written as indicies 1, and the largest value as indicies 2.
for example: x_pos = [1,2] is okay. --> x_pos[2,1] is NOT okay. 
    will be changed in newer versions.
"""
function place_wall(arr::Array{Int64, 3}, x_pos::Array{Float64, 1}, y_pos::Array{Float64, 1}, z_pos::Array{Float64, 1}, Δd::Float64)

    if x_pos[1]==x_pos[2]
        println("X pos is plane")
        for i in 1:size(arr,1)
            for j in Int(ceil(y_pos[1]/Δd)):Int(floor(y_pos[2]/Δd))
                for k in Int(ceil(z_pos[1]/Δd)):Int(floor(z_pos[2]/Δd))

                    B = arr[i,j,k]
                    
                    if (((i*Δd - x_pos[1] >= -Δd)) && ((i*Δd - x_pos[1]) <= 0))
                        X = 2
                        #println("*******")
                        #println("before: ", arr[i,j,k], "\t B: ", B)
                        arr[i,j,k] = merge(B,X)
                        #println(arr[i,j,k])
                    elseif (((i*Δd - x_pos[1]) <= Δd) && ((i*Δd - x_pos[1]) >= 0))
                        X = 1
                        #println("*******")
                        #println("before: ", arr[i,j,k], "\t B: ", B)
                        arr[i,j,k] = merge(B,X)
                        #println(arr[i,j,k])
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
                    if (((j*Δd - y_pos[1]) >= -Δd) && ((j*Δd - y_pos[1]) <= 0))
                        Y = 4
                        #println("*******")
                        #println("before: ", arr[i,j,k], "\t B: ", B)
                        arr[i,j,k] = merge(B,Y)
                        #println(arr[i,j,k])
                    elseif (((j*Δd - y_pos[1]) <= Δd) && ((j*Δd - y_pos[1]) >= 0))
                        Y = 3
                        #println("*******")
                        #println("before: ", arr[i,j,k], "\t B: ", B)
                        arr[i,j,k] = merge(B,Y)
                        #println(arr[i,j,k])
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
                   
                    if (((k*Δd - z_pos[1]) >= -Δd) && ((k*Δd - z_pos[1]) <= 0))
                        Z = 6
                        #println("*******")
                        #println("before: ", arr[i,j,k], "\t B: ", B)
                        arr[i,j,k] = merge(B,Z)
                        #println(arr[i,j,k])
                    elseif (((k*Δd - z_pos[1]) <= Δd) && ((k*Δd - z_pos[1]) >= 0))
                        Z = 5
                        #println("*******")
                        #println("before: ", arr[i,j,k], "\t B: ", B)
                        arr[i,j,k] = merge(B,Z)
                        #println(arr[i,j,k])

                    end
                end
            end
        end
    else
        println("Not a wall")
    end

    return arr
end



function place_wall2D(arr::Array{Int64, 2}, x_pos::Array{Float64, 1}, y_pos::Array{Float64, 1}, Δd::Float64)

    if x_pos[1]==x_pos[2]
        println("X pos is plane")
        for i in 1:size(arr,1)
            for j in Int(ceil(y_pos[1]/Δd)):Int(floor(y_pos[2]/Δd))
                B = arr[i,j]
                if (((i*Δd - x_pos[1] >= -Δd)) && ((i*Δd - x_pos[1]) <= 0))
                    X = 2
                    arr[i,j] = Merge(B,X)
                elseif (((i*Δd - x_pos[1]) <= Δd) && ((i*Δd - x_pos[1]) >= 0))
                    X = 1
                    arr[i,j] = Merge(B,X)
                end
            end
        end

        
    elseif y_pos[1] == y_pos[2]
        println("Y pos is plane")
        for i in Int(ceil(x_pos[1]/Δd)):Int(floor(x_pos[2]/Δd))
            for j in 1:size(arr,2)
                B = arr[i,j]
                if (((j*Δd - y_pos[1]) >= -Δd) && ((j*Δd - y_pos[1]) <= 0))
                    Y = 4
                    arr[i,j] = Merge(B,Y)
                elseif (((j*Δd - y_pos[1]) <= Δd) && ((j*Δd - y_pos[1]) >= 0))
                    Y = 3
                    arr[i,j] = Merge(B,Y) 
                end
            end
        end
    else
        println("Not a wall")
    end

    return arr
end
