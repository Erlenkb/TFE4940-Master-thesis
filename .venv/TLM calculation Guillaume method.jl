

"""
    calculate_pressure_matrix(labeled_tlm, SN, SE, SS, SW, SU, SD, IN, IE, IS, IW, IU, ID)

Computes the pressure matrix given by the scattering matrix and the labaled tlm to make sure what values to be used. 
"""










function propagate(
    Labeled_tlm::Array{Int64,3}, 
    SN::Array{Float64,3}, 
    SE::Array{Float64,3}, 
    SS::Array{Float64,3}, 
    SW::Array{Float64,3}, 
    SU::Array{Float64,3}, 
    SD::Array{Float64,3},
    IN::Array{Float64,3}, 
    IE::Array{Float64,3}, 
    IS::Array{Float64,3}, 
    IW::Array{Float64,3}, 
    IU::Array{Float64,3}, 
    ID::Array{Float64,3})



    for i in 1:size(Labeled_tlm,1)
        for j in 1:size(Labeled_tlm,2)
            for k in 1:size(Labeled_tlm,3)


                # Fluid
                if Labeled_tlm[i,j,k] == 0
                    IN[i,j,k] = SS[i - 1, j, k]
                    IE[i,j,k] = SW[i, j + 1, k]
                    IS[i,j,k] = SN[i + 1, j, k]
                    IW[i,j,k] = SE[i, j - 1, k]
                    IU[i,j,k] = SD[i, j, k + 1]
                    ID[i,j,k] = SU[i, j, k - 1]
                else
                    case = Labeled_tlm[i,j,k]
                    0 # Refl = R[1]

                    # Surfaces  --  6 cases
                    if case == 1
                        IN[i,j,k] = 0 # Refl * SN[i,j,k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 2
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = 0 # Refl * SS[i,j,k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 3
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = 0 # Refl * SW[i,j,k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 4
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = 0 # Refl * SE[i,j,k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 5
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = 0 # Refl * SD[i,j,k]
                    elseif case == 6
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = 0 # Refl * SU[i,j,k]
                        ID[i,j,k] = SU[i, j, k - 1]


                    # Edges  --  
                    elseif case == 12
                        IN[i,j,k] = 0 # Refl * SN[i,j,k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = 0 # Refl * SW[i,j,k] 
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 14
                        IN[i,j,k] = 0 # Refl * SN[i,j,k]
                        IE[i,j,k] = 0 # Refl * SE[i,j,k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 15
                        IN[i,j,k] = 0 # Refl * SN[i,j,k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = 0 # Refl * SD[i,j,k]
                    elseif case == 16
                        IN[i,j,k] = 0 # Refl * SN[i,j,k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = 0 # Refl * SU[i,j,k]
                        ID[i,j,k] = SU[i, j, k - 1]

                    elseif case == 23
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = 0 # Refl * SS[i,j,k]
                        IW[i,j,k] = 0 # Refl * SW[i,j,k] 
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 24
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = 0 # Refl * SE[i,j,k]
                        IS[i,j,k] = 0 # Refl * SS[i,j,k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 25
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = 0 # Refl * SS[i,j,k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = 0 # Refl * SD[i,j,k]
                    elseif case == 26
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = 0 # Refl * SS[i,j,k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = 0 # Refl * SU[i,j,k]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 35
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = 0 # Refl * SW[i,j,k] 
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = 0 # Refl * SD[i,j,k]
                    elseif case == 36
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = 0 # Refl * SW[i,j,k] 
                        IU[i,j,k] = 0 # Refl * SU[i,j,k]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 45
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = 0 # Refl * SE[i,j,k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = 0 # Refl * SD[i,j,k]
                    elseif case == 46
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = 0 # Refl * SE[i,j,k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = 0 # Refl * SU[i,j,k]
                        ID[i,j,k] = SU[i, j, k - 1]


                    # corners
                    elseif case == 135
                        IN[i,j,k] = 0 # Refl * SN[i,j,k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = 0 # Refl * SW[i,j,k] 
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = 0 # Refl * SD[i,j,k]
                    elseif case == 145
                        IN[i,j,k] = 0 # Refl * SN[i,j,k]
                        IE[i,j,k] = 0 # Refl * SE[i,j,k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = 0 # Refl * SD[i,j,k]
                    elseif case == 235
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = 0 # Refl * SS[i,j,k]
                        IW[i,j,k] = 0 # Refl * SW[i,j,k] 
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = 0 # Refl * SD[i,j,k]
                    elseif case == 245
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = 0 # Refl * SE[i,j,k]
                        IS[i,j,k] = 0 # Refl * SS[i,j,k]
                        IW[i,j,k] = SE[i, j - 1, k] 
                        IU[i,j,k] = SD[i, j, k + 1]
                        ID[i,j,k] = 0 # Refl * SD[i,j,k]
                    elseif case == 136
                        IN[i,j,k] = 0 # Refl * SN[i,j,k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = 0 # Refl * SW[i,j,k]
                        IU[i,j,k] = 0 # Refl * SU[i,j,k]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 146  
                        IN[i,j,k] = 0 # Refl * SN[i,j,k]
                        IE[i,j,k] = 0 # Refl * SE[i,j,k]
                        IS[i,j,k] = SN[i + 1, j, k]
                        IW[i,j,k] = SE[i, j - 1, k]
                        IU[i,j,k] = 0 # Refl * SU[i,j,k]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 236
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = SW[i, j + 1, k]
                        IS[i,j,k] = 0 # Refl * SS[i,j,k]
                        IW[i,j,k] = 0 # Refl * SW[i,j,k]
                        IU[i,j,k] = 0 # Refl * SU[i,j,k]
                        ID[i,j,k] = SU[i, j, k - 1]
                    elseif case == 246
                        IN[i,j,k] = SS[i - 1, j, k]
                        IE[i,j,k] = 0 # Refl * SE[i,j,k]
                        IS[i,j,k] = 0 # Refl * SS[i,j,k]
                        IW[i,j,k] = SE[i, j - 1, k] 
                        IU[i,j,k] = 0 # Refl * SU[i,j,k]
                        ID[i,j,k] = SU[i, j, k - 1]
                    end
                end
            end
        end
    end

    return IN, IE, IS, IW, IU, ID
end








"""
    calculate_scattering_matrix( SN, SE, SS, SW, SU, SD, IN, IE, IS, IW, IU, ID)

Computes the scattering matrix given the pressure matrix.
"""


function scattering(
    IN::Array{Float64,3}, 
    IE::Array{Float64,3}, 
    IS::Array{Float64,3}, 
    IW::Array{Float64,3}, 
    IU::Array{Float64,3}, 
    ID::Array{Float64,3})
    SW = (1/3) .* (-2*IW .+ IN .+ IE .+ IS .+ IU .+ ID)
    SN = (1/3) .* (IW .+ (-2*IN) .+ IE .+ IS .+ IU .+ ID)
    SE = (1/3) .* (IW .+ IN .+ (-2*IE) .+ IS .+ IU .+ ID)
    SS = (1/3) .* (IW .+ IN .+ IE .+ (-2*IS) .+ IU .+ ID)
    SU = (1/3) .* (IW .+ IN .+ IE .+ IS .+ (-2*IU) .+ ID)
    SD = (1/3) .* (IW .+ IN .+ IE .+ IS .+ IU .+ (-2*ID))

    return SN, SE, SS, SW, SU, SD

end




