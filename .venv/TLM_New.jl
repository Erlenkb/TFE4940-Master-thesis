using FFTW, Plots
using ProgressMeter
using DSP
using Statistics, StatsPlots


include("ltau.jl")




##### Global variables


Δd = 0.2  # in meters
tot_time = 1.0   # in seconds
energy_time = 0.2  # in seconds
freq = 50
l = 14.0
width = 3.0
height = 2.5

impulse = false
harmonic = true
sweep = false
source_position = [1.2,1.1,1.5]

R = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]





##### Instanciate global variables

runtime = true
Temp = 291
ρ_air = 1.225
c = 343.2*sqrt(Temp/293)
Δt = (Δd / c)
fs = Int(1÷Δt)
p0 = 2e-5
N = Int(tot_time ÷ Δt)
nx = Int(l ÷ Δd + 1)
ny = Int(width ÷ Δd + 1)
nz = Int(height ÷ Δd + 1)
meas_pos1 = [1.0, 1.0, 1.0]
meas_pos2 = [0.5, 1.2, 1.2]
meas_pos3 = [2.2, 1.2, 1.2]
mic1 = []
mic2 = []
mic3 = []
mic1_Lp = []
mic2_Lp = []
mic3_Lp = []

Labeled_tlm = zeros(Int, (nx,ny,nz))
pressure_grid = [] 
SN = zeros((nx,ny,nz))
SE = zeros((nx,ny,nz))
SS = zeros((nx,ny,nz))
SW = zeros((nx,ny,nz))
SU = zeros((nx,ny,nz))
SD = zeros((nx,ny,nz))
IN = zeros((nx,ny,nz))
IE = zeros((nx,ny,nz))
IS = zeros((nx,ny,nz))
IW = zeros((nx,ny,nz))
IU = zeros((nx,ny,nz))
ID = zeros((nx,ny,nz))

println(fs)

f(x) = 50 + cos(x*2*pi*600)


function plot_fft(signal, fs, start)
    
    start_ind = Int(start ÷ Δt)
    end_idx = N  # Assume end index is the length of the signal array

    # Slice the signal array from start to end
    sliced_signal = signal[start_ind:end_idx]
    N_FFT = length(sliced_signal)
    # Compute the FFT
    fft_result = fft(sliced_signal)

    # Compute frequency axis
    df = fs / (N_FFT - 1)
    freq_axis = 0:df:fs

    # Find the index corresponding to the Nyquist frequency
    nyquist_idx = div(N_FFT, 2) + 1

    maxarg = argmax(20 * log10.(abs.(fft_result[1:nyquist_idx])))

    # Plot the magnitude spectrum (positive frequencies only)
    #Plots.plot(freq_axis[1:nyquist_idx], 20 * log10.(abs.(fft_result[1:nyquist_idx])), xlims=(nyquist_idx,nothing),xaxis=:log10, xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", legend=false, xlim=(0, fs/2))
    return maxarg 
end










function impulse_val(it)
    return 
end

function harmonic_val(it)
    return 40*cos(it*Δt*2*pi*freq)
end

function sweep_val(it)

end


function _real_to_index(val)
    ind = [Int(div(val[1],Δd)), Int(div(val[2],Δd)), Int(div(val[3],Δd))]
    return ind
end


function tlmGrid()
    println("Creating ShoeLabeled_tlm shape")
    println("Δd: ",Δd, " m")
    println("size:\t x_direction: ", nx*Δd, " m \t y_direction: ",ny*Δd," m \t z_direction: ", nz*Δd, " m")

    # Set surface values by checking the position
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz

                if i == 1
                    Labeled_tlm[i, j, k] = 1
                elseif i == nx
                    Labeled_tlm[i, j, k] = 2
                elseif j == 1
                    Labeled_tlm[i, j, k] = 3
                elseif j == ny
                    Labeled_tlm[i, j, k] = 4
                elseif k == 1
                    Labeled_tlm[i, j, k] = 5
                elseif k == nz
                    Labeled_tlm[i, j, k] = 6
                end
            end
        end
    end

    # Set Edge values
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz

                if i == 1 && j == ny
                    Labeled_tlm[i, j, k] = 14
                elseif j == ny && k == 1
                    Labeled_tlm[i, j, k] = 45
                elseif i == nx && j == ny
                    Labeled_tlm[i, j, k] = 24
                elseif j == ny && k == nz
                    Labeled_tlm[i, j, k] = 46
                elseif i == nx && k == 1
                    Labeled_tlm[i, j, k] = 25
                elseif i == nx && k == nz
                    Labeled_tlm[i, j, k] = 26
                elseif i == nx && j == 1
                    Labeled_tlm[i, j, k] = 23
                elseif j == 1 && k == 1
                    Labeled_tlm[i, j, k] = 35
                elseif i == 1 && k == 1
                    Labeled_tlm[i, j, k] = 15
                elseif i == 1 && k == nz
                    Labeled_tlm[i, j, k] = 16
                elseif i == 1 && j == 1
                    Labeled_tlm[i, j, k] = 13
                elseif j == 1 && k == nz
                    Labeled_tlm[i, j, k] = 36
                end
            end
        end
    end

    # Set Corner values
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz

                if i == nx && j == 1 && k == 1
                    Labeled_tlm[i, j, k] = 235
                elseif i == 1 && j == 1 && k == 1
                    Labeled_tlm[i, j, k] = 135
                elseif i == 1 && j == ny && k == 1
                    Labeled_tlm[i, j, k] = 145
                elseif i == nx && j == ny && k == 1
                    Labeled_tlm[i, j, k] = 245
                elseif i == nx && j == 1 && k == nz
                    Labeled_tlm[i, j, k] = 236
                elseif i == 1 && j == 1 && k == nz
                    Labeled_tlm[i, j, k] = 136
                elseif i == 1 && j == ny && k == nz
                    Labeled_tlm[i, j, k] = 146
                elseif i == nx && j == ny && k == nz
                    Labeled_tlm[i, j, k] = 246
                end
            end
        end
    end
    return
end


function calculate_next_step(it)
    
    # Insert energy into 
    update_incident_with_energy(it)

   

    # Update scatter values
    scatter()
    
    # Update incident pulses
    propagate()
    

end

function meas_pressure()

    for item in pressure_grid
        push!(mic1, item[_real_to_index(meas_pos1)[1],_real_to_index(meas_pos1)[2],_real_to_index(meas_pos1)[3]])
        push!(mic2, item[_real_to_index(meas_pos2)[1],_real_to_index(meas_pos2)[2],_real_to_index(meas_pos2)[3]])
        push!(mic3, item[_real_to_index(meas_pos3)[1],_real_to_index(meas_pos3)[2],_real_to_index(meas_pos3)[3]])
    end
end






function run_simulation()
    prog1 = Progress(N)
    for it in 1:N
       
        # Caculate values for next timestep
        calculate_next_step(it)

        # Sum overall pressure for each node
        overal_pressure()
        next!(prog1)
        #readline()
    end
    meas_pressure()
    #println(plot_fft(mic3, fs, energy_time))
    plot_data(ltau(mic1, fs, Δt * 10, p0), ltau(mic2, fs, Δt * 10, p0), ltau(mic3, fs, Δt * 10, p0))


end








function update_incident_with_energy(it)
    if Δt*it >= energy_time

        return
    end
    if impulse == true
        insert_energy(impulse_val(it))
    end
    if harmonic == true
        insert_energy(harmonic_val(it))
    end
    if sweep == true
        insert_energy(sweep_val(it))
    end
end

function insert_energy(val)
    source_pos = _real_to_index(source_position)
    val  = val * 10
    IN[source_pos[1]+1,source_pos[2],source_pos[3]] = val
    IE[source_pos[1],source_pos[2]-1,source_pos[3]] = val
    IS[source_pos[1]-1,source_pos[2],source_pos[3]] = val
    IW[source_pos[1],source_pos[2]+1,source_pos[3]] = val
    IU[source_pos[1],source_pos[2],source_pos[3]-1] = val
    ID[source_pos[1],source_pos[2],source_pos[3]+1] = val
end

function overal_pressure()
    sum_nodes = ((1/3)*(IN .+ IE .+ IS .+ IW .+ IU .+ ID))
    push!(pressure_grid, sum_nodes)
    return
end

function scatter()
    global SW, SN, SE, SS, SU, SD
    SW = (1/3) * ((-2*IW) .+ IN .+ IE .+ IS .+ IU .+ ID)
    SN = (1/3) * (IW .+ (-2*IN) .+ IE .+ IS .+ IU .+ ID)
    SE = (1/3) * (IW .+ IN .+ (-2*IE) .+ IS .+ IU .+ ID)
    SS = (1/3) * (IW .+ IN .+ IE .+ (-2*IS) .+ IU .+ ID)
    SU = (1/3) * (IW .+ IN .+ IE .+ IS .+ (-2*IU) .+ ID)
    SD = (1/3) * (IW .+ IN .+ IE .+ IS .+ IU .+ (-2*ID))
    
    return
end

function case(n::Int64)
    Diffusor = false
    if n < 0
        Diffusor = true
        n = abs(n)
    end
    num_str = string(n)
    used_nums = [parse(Int, digit) for digit in num_str]
    all_nums = Set(1:6)
    unused_nums = setdiff(collect(all_nums), used_nums)

    return used_nums, unused_nums, Diffusor
end

function propagate()
    for i in 1:size(Labeled_tlm,1)
        for j in 1:size(Labeled_tlm,2)
            for k in 1:size(Labeled_tlm,3)
                #println("************************")
                if Labeled_tlm[i,j,k] != 0
                    
                    case_used, case_unused, diffusor = case(Labeled_tlm[i,j,k])
                    
                    for n in case_used
                        Refl = diffusor ? R[6] : R[n]

                        #println("n: ", n, "\t case_used: ", case_used, "\t case_unused: ",case_unused, )
                        if n == 1
                            
                            IN[i,j,k] =  Refl * SN[i,j,k]
                            
                            #println("reflected north")                  
                        elseif n == 2
                            IS[i,j,k] =  Refl * SS[i,j,k]
                            #println("reflected south")    
                        elseif n == 3
                            IW[i,j,k] = Refl * SW[i,j,k]
                            #println("reflected west")    
                        elseif n == 4
                            IE[i,j,k] = Refl * SE[i,j,k]
                            #println("reflected east")    
                        elseif n == 5
                            ID[i,j,k] = Refl * SD[i,j,k]
                            #println("reflected down")    
                        elseif n == 6
                           IU[i,j,k] = Refl * SU[i,j,k] 
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
    return
end

function plot_data(arr1, arr2, arr3)
    println("Plotting arrays")
    t = collect(0:N-1) / fs

    sos = f.(t)
    #display(arr1)
    Plots.plot(t, arr1, label="Mic 1")
    Plots.plot!(t, arr2, label="Mic 2")
    Plots.plot!(t, arr3, label="Mic 3")
    #Plots.plot!(t, sos, label="600 Hz")
    xlabel!("Time (s)")
    ylabel!("Magnitude (dB)")
    title!("Time weighted sound pressure \nlevel at three positions")
    #ylims!((0,100))
    #xlims!(0.25,0.35)

end


function main()
    tlmGrid()
    #display(Labeled_tlm)
    run_simulation()
end


main()

