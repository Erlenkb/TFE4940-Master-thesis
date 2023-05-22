
"""
    generate_noise_signal(N, t, dB)

Computes the noise signal of length 'N' for a given time 't' and the desired SPL 'dB'
"""


function generate_noise_signal(N, t::Float64, dB)
    pressure = 10^(dB/20) * 20e-6  # calculate the pressure level from dB
    return randn(N) .* sqrt(t) .* pressure
end




"""
    pressure_value(t)

computes the pressure value for the given time step 't'
"""


function pressure_value(t::Float64)
    return signal_strength_Pa*cos(t*2*pi*freq)
end



"""
    pressure_value_directional(t)

computes the pressure value for a given time step 't' with a directivity facotor
"""


function pressure_value_directional(t::Float64)
    return signal_strength_Pa*sin(t*2*pi*freq)*directivity_factor
end




"""
    overal_pressure(IN, IE, IS, IW, IU, ID)

computes the overall pressure inside the pressure grid given by the 6 transmission lines.
"""


function overal_pressure(
    IN::Array{Float64,3}, 
    IE::Array{Float64,3},
    IS::Array{Float64,3}, 
    IW::Array{Float64,3}, 
    IU::Array{Float64,3}, 
    ID::Array{Float64,3})
    P_grid = (1/3) .* (IW .+ IN .+ IE .+ IS .+ IU .+ ID)
    return P_grid
end




"""
    _Lp(arr)

Computes the sound pressure level for a given array containing sound pressure values.
return the 20*log(p/p0)
"""


function _Lp(arr)
    return 20 .* log10.(arr / p0)
end





"""
    Leq_fast(arr, fs)

Computes the L_eq,fast value for a pressure array 'arr' given the sampling frequency 'fs'
"""


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





"""
    time_to_discrete(timeval,fs)

computes the discrete value of the timeval
"""


function time_to_discrete(timeval::Float64, fs::Int64)
    return Int(timeval * fs)
end




"""
    check_fs(fs)
verify if the given distance between each node satisfy 10 points per waelength
"""


function check_fs(fs)
    if(fs < (0.1*c/freq))
        println("Samplingfrequency is to low")
    else
        println("Samplingfrequency is sufficiently high")
    end
end
