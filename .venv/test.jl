using SignalAnalysis
using Plots
using DSP
fs = 10000


A = zeros((3,3,3))
B = zeros((3,3))





function test1()
  for i in 1:3
    for j in 1:3
      B[i,j] = i*j
    end
  end
  return
end


function test2()
  
  for i in 2:3
    A[:,:,i] = B
  end
  C = A
  global B = A[:,:,2].+C[:,:,2]
  return
end


function test3()
  test1()
  test2()
  display(A)
  
  test2()
  display(A)
  return

end

test3()











"""
function _generate_sweep(start_freq, end_freq, duration, fs)
  return chirp(start_freq, end_freq, duration, fs, window=(tukey, 0.05); shape=:linear)
end



sweep_signal = _generate_sweep(100, 10000, 0.1, fs)
display(sweep_signal)
#Plots.plot(real(sweep_signal),imag(sweep_signal))

fs = 22000



ts = range(0, stop=6, step=1/fs)
signal = @. sin(2π*ts^2)

function generateSweep(start_freq, end_freq, duration, sample_rate)
  t = collect(0:1/sample_rate:duration)
  freq_range = end_freq - start_freq
  phase = 2π * (start_freq * t + 0.5 * freq_range * t.^2 / duration)
  sweep_signal = sin.(phase)
  
  window = tukey(length(sweep_signal), 0.05)  # Tukey window with alpha = 0.1
  sweep_signal = sweep_signal .* window
  
  return sweep_signal, t
end

sweep, t = generateSweep(10, 500, 0.6, fs)

Plots.plot(t, sweep)
"""