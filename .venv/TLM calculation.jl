

"""
    calculate_pressure_matrix(labeled_tlm, SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)

Computes the pressure matrix given by the scattering matrix and the labaled tlm to make sure what values to be used. 
"""

function calculate_pressure_matrix(
    Labeled_tlm::Array{Int64,3}, 
    SN::Array{Float64,3}, 
    SE::Array{Float64,3}, 
    SS::Array{Float64,3}, 
    SW::Array{Float64,3}, 
    SU::Array{Float64,3}, 
    SD::Array{Float64,3},
    PN::Array{Float64,3}, 
    PE::Array{Float64,3}, 
    PS::Array{Float64,3}, 
    PW::Array{Float64,3}, 
    PU::Array{Float64,3}, 
    PD::Array{Float64,3})



    for i in 1:size(Labeled_tlm,1)
        for j in 1:size(Labeled_tlm,2)
            for k in 1:size(Labeled_tlm,3)

                if Labeled_tlm[i,j,k] != 0

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

                else
                    PN[i,j,k] = SS[i - 1, j, k]
                    PE[i,j,k] = SW[i, j + 1, k]
                    PS[i,j,k] = SN[i + 1, j, k]
                    PW[i,j,k] = SE[i, j - 1, k]
                    PU[i,j,k] = SD[i, j, k + 1]
                    PD[i,j,k] = SU[i, j, k - 1]

                end
            end
        end
    end

    return SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD
end








"""
    calculate_scattering_matrix( SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD)

Computes the scattering matrix given the pressure matrix.
"""


function calculate_scattering_matrix(
    SN::Array{Float64,3}, 
    SE::Array{Float64,3}, 
    SS::Array{Float64,3}, 
    SW::Array{Float64,3}, 
    SU::Array{Float64,3}, 
    SD::Array{Float64,3},
    PN::Array{Float64,3}, 
    PE::Array{Float64,3}, 
    PS::Array{Float64,3}, 
    PW::Array{Float64,3}, 
    PU::Array{Float64,3}, 
    PD::Array{Float64,3})
    SW = (1/3) .* ((-2*PW) .+ PN .+ PE .+ PS .+ PU .+ PD)
    SN = (1/3) .* (PW .+ (-2*PN) .+ PE .+ PS .+ PU .+ PD)
    SE = (1/3) .* (PW .+ PN .+ (-2*PE) .+ PS .+ PU .+ PD)
    SS = (1/3) .* (PW .+ PN .+ PE .+ (-2*PS) .+ PU .+ PD)
    SU = (1/3) .* (PW .+ PN .+ PE .+ PS .+ (-2*PU) .+ PD)
    SD = (1/3) .* (PW .+ PN .+ PE .+ PS .+ PU .+ (-2*PD))

    return SN, SE, SS, SW, SU, SD, PN, PE, PS, PW, PU, PD

end




