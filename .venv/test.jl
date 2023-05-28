using Plots

function logarithmic_sweep(start_freq, end_freq, total_duration, sample_rate)
    start_freq = max(start_freq, 1e-15)  # Avoid negative or zero frequencies
    end_freq = max(end_freq, 1e-15)
    
    num_samples = Int(total_duration * sample_rate)
    t = (0:num_samples-1) / sample_rate
    
    start_freq_log = log10(start_freq)
    end_freq_log = log10(end_freq)
    
    frequencies = 10 .^ (start_freq_log .+ (end_freq_log - start_freq_log) .* t ./ total_duration)
    phases = 2π * cumsum(frequencies) / sample_rate
    sweep = sin.(phases)
    
    return sweep, length(sweep)
end


