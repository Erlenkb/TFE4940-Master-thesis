

"Express an adimensional ratio in decibels"
db(x)=10log10(x)

"""
    ltau(p,fs,τ,pref=p0())

Compute, for a pressure signal `p` sampled at `fs` Hz, the exponential time-weighted sound pressure 
level with time constant `τ`` (s). The output is a vector whose length equals that of `p`. As a rule 
of thumb, the output from `t=0` to `t=τ/2`. should be discarded. `
The unit of the result is dB re `pref`.  
"""
function ltau(p,fs::Int64,τ::Real,pref::Real=p0())
	bufsize=convert(Int64,floor(0.5*τ*fs)) #number of samples to extrapolate
	p2=p.^2                                #squaring the signals
	#fictitious samples are added at the beginning to avoid very negative sound levels right after t=0
	p2=vcat(mean(p2[1:bufsize])*ones(bufsize),p2)  
	#Implementing exponential time weighting as a low pass filter
	responsetype = Lowpass(1/(2π*τ); fs=fs)    #2pi factor inserted when tuning the decay rate.
	designmethod = Butterworth(1)
	htau=digitalfilter(responsetype, designmethod) #As of 28.10.2022 analogfilter() fails
	y=filt(htau,p2)                        #filtering
	y=1/(pref^2)*y[bufsize+1:end]          #adjusting the length to that of the input vector
	l=db.(y)
	return l
end