using Plots
using Printf
using DSP
using Statistics, StatsPlots

function merge(A::Int64, B::Float64)
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

function merge(A::Int, B::Int)::Int
    # Concatenate A and B as strings
    str = string(A, B)

    # Remove duplicates and sort the remaining characters
    unique_chars = sort(unique(str))

    # Convert the sorted string back to an integer and return it
    result = 0
    for c in unique_chars
        result = result * 10 + (c - '0')
    end
    return result
end



"Express an adimensional ratio in decibels"
db(x)=10log10(x)

"""
    ltau(p,fs,τ,pref=p0())

Compute, for a pressure signal `p` sampled at `fs` Hz, the exponential time-weighted sound pressure 
level with time constant `τ`` (s). The output is a vector whose length equals that of `p`. As a rule 
of thumb, the output from `t=0` to `t=τ/2`. should be discarded. `
The unit of the result is dB re `pref`.  
"""
function ltau(p,fs::Int64,τ::Real,pref)
	bufsize=convert(Int64,floor(0.5*τ*fs)) #number of samples to extrapolate
	p2=p.^2                                #squaring the signals
	#fictitious samples are added at the beginning to avoid very negative sound levels right after t=0
	p2=vcat(mean(p2[1:bufsize])*ones(bufsize),p2)  
    println(p2)
    
	#Implementing exponential time weighting as a low pass filter
	responsetype = Lowpass(1/(2π*τ); fs=fs)    #2pi factor inserted when tuning the decay rate.
	designmethod = Butterworth(1)
	htau=digitalfilter(responsetype, designmethod) #As of 28.10.2022 analogfilter() fails
	y=filt(htau,p2)                        #filtering
	y=1/(pref^2)*y[bufsize+1:end]          #adjusting the length to that of the input vector
	l=db.(y)
	return l
end





T = 1
N = 100
freq = 20
time = range(0, stop=T, length=N)
f(x) = 1*sin(x*2*pi*freq)
pressure_arr = []


for n in 1:N
  if n < 20
    push!(pressure_arr, f(n/N))
  else
    push!(pressure_arr, 0)
  end
end

p = ltau(abs.(pressure_arr), 100, 0.02, 20*10^(-6))

plot(
    plot(time, pressure_arr, label="Pressure values"),
    plot(time, p, label="Time weighted sound level"),
    layout=(1,2),
    size=(800,400),
)




