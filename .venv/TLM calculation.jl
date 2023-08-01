
function Reflection(Z_a, Z_T)
    Γ = (Z_a - Z_T) / (Z_a + Z_T)
    Re = ((1+Γ) - sqrt(2)*(1-Γ)) / ((1+Γ) + sqrt(2)*(1-Γ))
    return Re
end






"""
    calculate_pressure_matrix(labeled_tlm, SN, SE, SS, SW, SU, SD, IN, IE, IS, IW, IU, ID)

Computes the pressure matrix given by the scattering matrix and the labaled tlm to make sure what values to be used. 
"""




reflection = [0.17,0.17,0.17,0.17,0.17,0.17,0.17]#[0.17,0.17,0.17,0.17,0.17,0.17,0.17]#[Reflection(R[x],Z_T) for x in 1:6]
#R = [0.17,0.17,0.17,0.17,0.17,0.17,0.17]
R = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
reflection = R

println(reflection)
function propagate(
    Labeled_tlm::Array{Int64,3}, 
    SN::Array{Float64,3}, 
    SE::Array{Float64,3}, 
    SS::Array{Float64,3}, 
    SW::Array{Float64,3}, 
    SU::Array{Float64,3}, 
    SD::Array{Float64,3})

    

    for i in 1:size(Labeled_tlm,1)
        for j in 1:size(Labeled_tlm,2)
            for k in 1:size(Labeled_tlm,3)
                #println("************************")
                if Labeled_tlm[i,j,k] != 0
                    
                    case_used, case_unused, diffusor = case(Labeled_tlm[i,j,k])
                    
                    for n in case_used
                        Refl = diffusor ? reflection[6] : reflection[n]

                        #println("n: ", n, "\t case_used: ", case_used, "\t case_unused: ",case_unused, )
                        if n == 1
                            if (abs(Refl * SN[i,j,k]) < 1e-11) IN[i,j,k] = 0
                            else IN[i,j,k] =  Refl * SN[i,j,k] end
                            #println("reflected north")                  
                        elseif n == 2
                            if (abs(Refl * SS[i,j,k]) < 1e-11) IS[i,j,k] = 0
                            else IS[i,j,k] =  Refl * SS[i,j,k] end
                            #println("reflected south")    
                        elseif n == 3
                            if (abs(Refl * SW[i,j,k]) < 1e-11) IW[i,j,k] = 0
                            else IW[i,j,k] = Refl * SW[i,j,k] end
                            #println("reflected west")    
                        elseif n == 4
                            if (abs(Refl * SE[i,j,k]) < 1e-11) IE[i,j,k] = 0
                            else IE[i,j,k] = Refl * SE[i,j,k] end
                            #println("reflected east")    
                        elseif n == 5
                            if (abs(Refl * SD[i,j,k]) < 1e-11) ID[i,j,k] = 0
                            else ID[i,j,k] = Refl * SD[i,j,k] end
                            #println("reflected down")    
                        elseif n == 6
                            if (abs(Refl * SU[i,j,k]) < 1e-11) IU[i,j,k] = 0
                            else IU[i,j,k] = Refl * SU[i,j,k] end
                            #println("reflected up")    
                        end
                    end

                    for n in case_unused
                        if n == 1
                            IN[i,j,k] = SS[i - 1, j, k]

                            #println("not reflected north")
                        elseif n == 2
                            IS[i,j,k] = SN[i + 1, j, k]
                            #println("not reflected south")
                        elseif n == 3
                            IW[i,j,k] = SE[i, j - 1, k]
                            #println("not reflected west")
                        elseif n == 4
                            IE[i,j,k] = SW[i, j + 1, k]
                            #println("not reflected east")
                        elseif n == 5
                            ID[i,j,k] = SU[i, j, k - 1]
                            #println("not reflected down")
                        elseif n == 6
                            IU[i,j,k] = SD[i, j, k + 1]
                            #println("not reflected up")
                        end
                    end
                else
                    #println("Did the fluid")
                    IN[i,j,k] = SS[i - 1, j, k]
                    IE[i,j,k] = SW[i, j + 1, k]
                    IS[i,j,k] = SN[i + 1, j, k]
                    IW[i,j,k] = SE[i, j - 1, k]
                    IU[i,j,k] = SD[i, j, k + 1]
                    ID[i,j,k] = SU[i, j, k - 1]
                end
                #println(IN[i,j,k])
                #println(IE[i,j,k])
                #println(IS[i,j,k])
                #println(IW[i,j,k])
                #println(IU[i,j,k])
                #println(ID[i,j,k])
            end
        end
    end

    return IN, IE, IS, IW, IU, ID
end


function propagate2(
    Labeled_tlm::Array{Int64,3}, 
    SN::Array{Float64,3}, 
    SE::Array{Float64,3}, 
    SS::Array{Float64,3}, 
    SW::Array{Float64,3}, 
    SU::Array{Float64,3}, 
    SD::Array{Float64,3})
    for i in 1:size(Labeled_tlm,1)
        for j in 1:size(Labeled_tlm,2)
            for k in 1:size(Labeled_tlm,3)

                case_used, case_unused, diffusor = case(Labeled_tlm[i,j,k])
                for n in case_used
                    if n == 0 Refl = 0 else Refl = diffusor ? R[6] : R[n] end

                    IN[i,j,k] = (1 in case_used) ? Refl * SN[i,j,k] : SS[i - 1, j, k] 
                    IS[i,j,k] = (2 in case_used) ? Refl * SS[i,j,k] : SN[i + 1, j, k] 
                    IW[i,j,k] = (3 in case_used) ? Refl * SW[i,j,k] : SE[i, j - 1, k] 
                    IE[i,j,k] = (4 in case_used) ? Refl * SE[i,j,k] : SW[i, j + 1, k] 
                    ID[i,j,k] = (5 in case_used) ? Refl * SD[i,j,k] : SU[i, j, k - 1] 
                    IU[i,j,k] = (6 in case_used) ? Refl * SU[i,j,k] : SD[i, j, k + 1] 
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

    SW = (1/3) * ((-2*IW) .+ IN .+ IE .+ IS .+ IU .+ ID)
    SN = (1/3) * (IW .+ (-2*IN) .+ IE .+ IS .+ IU .+ ID)
    SE = (1/3) * (IW .+ IN .+ (-2*IE) .+ IS .+ IU .+ ID)
    SS = (1/3) * (IW .+ IN .+ IE .+ (-2*IS) .+ IU .+ ID)
    SU = (1/3) * (IW .+ IN .+ IE .+ IS .+ (-2*IU) .+ ID)
    SD = (1/3) * (IW .+ IN .+ IE .+ IS .+ IU .+ (-2*ID))

    return SN, SE, SS, SW, SU, SD

end




