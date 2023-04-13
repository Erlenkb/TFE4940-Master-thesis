import matplotlib.pyplot as plt
import numpy as np


####### Parameterverdier ######
tykkelse_kryssfiner = 0.02
tykkelse_absorbent_bunn_15mm = 0.015
tykkelse_absorbent_bunn_20mm = 0.02
f_0Box = 400
f_nedregrense = 250
f_øvregrense = 10000
N = 13
N_it = 2
b_box = 0.30
d_box = 0.25



##############################
### Dybde =< 25 cm
### Bredde =< 30 cm

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
#######################################

def _frekvens_av_dybde_MLS(d):
    f_0 = 86 / d
    f_øvre = 1.4 * f_0
    f_nedre = 0.7 * f_0
    return f_nedre, f_0, f_øvre

def _parametere_QRD(d_box, tykkelse_bunn, b_box, tykkelse_kant, N):
    d_maks = round(d_box/2 - tykkelse_bunn,3)
    b_QRD = round((b_box - 2*tykkelse_kant) / N,3)    
    print("d_maks,QRD <= ",d_maks, " m\t b_QRD = ", b_QRD," m")
    return d_maks, b_QRD

def _parametere_frekvens(f_0Box, f_nedregrense, f_øvregrense):
    if (f_nedregrense < f_0Box * 0.7):
        f_0Box = round(f_nedregrense / 0.7,0)
        print("Nedre grense ikke møtt. Erstatter f_0,Box til: ",f_0Box," Hz")
    else: print("Nedre grense tilfredsstiller f_0. \nTeoretiske f_nedregrense relativt til valgte nedre grense er:\nf_teoretiske,nedre = ",f_0Box * 0.7," Hz \t f_nedre_satt = ",f_nedregrense, " Hz\n")
    
    d = round(86/f_0Box,3)
    b = d * 2
    l = round(b * 2,2)
    print("d_box = ",d, " m\t b_box = ", b," m\tl_box ",l," m")
    
    return d, b, l

def _elementnummer_dybde_og_f_0(N, d_maks):
    n = []
    for m in range(N):
        n.append(m**2 % N)
    n_maks = max(n)
    
    f_0_QRD = round((172 * n_maks) / (N * d_maks),1)
    scaling_factor = d_maks / n_maks
    
    
    print("f_0_QRD = ",f_0_QRD, " Hz \nFølgende dybder per celle i meter er:\n")
    print("Element (m) \t depth [n] \t depth [cm]")
    for i, val in enumerate(n):
        print(i, "\t\t",val,"\t\t", round(val * scaling_factor * 100,1))
    print("\n")
    return n_maks, f_0_QRD, n

def _QRD_depth_calc(N, f_nedre):
    n = []
    for m in range(N):
        n.append(m**2 % N)
    f0 = f_nedre / 0.5
    n_maks = max(n)
    d_maks = round((172*n_maks)/(N*f0),2)
    
    print("f0: ", f0, " Hz\t d_maks: ",d_maks," m\t f_nedre: ",f_nedre, " Hz")
    return d_maks

def _QRD_width_check(b_QRD, N):
    print("Nedre frekvensgrense for b*N er: ", round(344/(N*b_QRD),1), " Hz")
    return 1

def _QRD_øvre_frekvens(b_QRD): 
    print("Øvre grensefrekvens for en QRD med bredde ",b_QRD," m er ",round(573 / b_QRD,1)," Hz")
    return 573 / b_QRD

def _plot_Box_shape_template(b,d_maks,b_QRD,N,tykkelse_kant=tykkelse_kryssfiner, tykkelse_bunn=tykkelse_kryssfiner):
    n = []
    n_cm = []
    for m in range(N*2):
        n.append(m**2 % N)
    
    n_maks = max(n)
    for i in n: n_cm.append(i*(d_maks / n_maks))
    n_cm = n_cm[(N+1)//2:N+N//2+2]
    
    x_steps = np.linspace(tykkelse_kant, tykkelse_kant+((N)*b_QRD), N+1)

    for i,val in enumerate(n_cm): n_cm[i] = d_maks - val

    fig, ax = plt.subplots(figsize=(8,7))
    
    
    
    ax.step(x_steps, n_cm, where="post", label="Spalter som utgjør QRD delen inne i boksen")
    ax.vlines([0,tykkelse_kant,tykkelse_kant+N*b_QRD,2*tykkelse_kant+N*b_QRD],0,d_maks,colors="black")
    ax.hlines([d_maks,d_maks],[0,tykkelse_kant+N*b_QRD],[tykkelse_kant,2*tykkelse_kant+N*b_QRD],colors="black")
    ax.vlines([0, 2*tykkelse_kant+N*b_QRD],0,-tykkelse_bunn,colors="black")
    ax.hlines([0,-tykkelse_bunn],0,2*tykkelse_kant+N*b_QRD,colors="black",label="Snitt av boks - kryssfiner")
    
    ax.grid()
    ax.legend()
    ax.set_xlim(-1.2*tykkelse_kant, 0.2*tykkelse_kant+2*tykkelse_kant+N*b_QRD)
    ax.set_ylim(-1.2*tykkelse_kant, 0.2*tykkelse_kant+2*tykkelse_kant+N*b_QRD)
    ax.set_title("Snitt av QRD del for diffusjonsbokser med N={0}".format(N))
    ax.set_ylabel("Dybde [m]")
    ax.set_xlabel("Bredde [m]")
    plt.show()
   
    return 1


def _print_frequency_with_QRD(d_box, b_box, tykkelse_absorbent):
    
    ##### Boks parametere
    box_f_nedre, box_f_0, box_f_øvre = _frekvens_av_dybde_MLS(d_box)
    
    ##### QRD parametere 1
    N_test = [7,11,13,17,21]
    n_maks = np.zeros(len(N_test))
    f_0_QRD = np.zeros(len(N_test))
    n = np.zeros(len(N_test))
    d_QRD_maks = np.zeros(len(N_test))
    b_QRD = np.zeros(len(N_test))
    QRD_f_nedre_verifisering = np.zeros(len(N_test))
    QRD_f_nedre = np.zeros(len(N_test))
    QRD_f_øvre = np.zeros(len(N_test))
    tykkelse_bunn = tykkelse_kryssfiner + tykkelse_absorbent_bunn_15mm
    for i, val in enumerate(N_test):
        
        d_QRD_maks[i], b_QRD[i] = _parametere_QRD(d_box, tykkelse_bunn, b_box, tykkelse_kryssfiner, val)
        n_maks[i], f_0_QRD[i], n = _elementnummer_dybde_og_f_0(val,d_QRD_maks[i])
        QRD_f_nedre_verifisering[i] = 344 / (val * b_QRD[i])
        QRD_f_nedre[i] = 0.5 * f_0_QRD[i]
        QRD_f_øvre[i] = 573 / b_QRD[i]

    
    n = []
    n_cm = []
    for m in range(N*2):
        n.append(m**2 % N)
    
    n_maks = max(n)
    for i in n: n_cm.append(i*(d_QRD_maks[0] / n_maks))
    n_cm = n_cm[(N+1)//2:N+N//2+2]
    
    x_steps = np.linspace(tykkelse_kryssfiner, tykkelse_kryssfiner+((N)*b_QRD[N_it]), N+1)

    for i,val in enumerate(n_cm): n_cm[i] = d_QRD_maks[0] - val


    ##### QRD parametere 2
    N_test = [7,11,13,17,21]
    n_maks_1 = np.zeros(len(N_test))
    f_0_QRD_1 = np.zeros(len(N_test))
    n_1 = np.zeros(len(N_test))
    d_QRD_maks_1 = np.zeros(len(N_test))
    b_QRD_1 = np.zeros(len(N_test))
    QRD_f_nedre_verifisering_1 = np.zeros(len(N_test))
    QRD_f_nedre_1 = np.zeros(len(N_test))
    QRD_f_øvre_1 = np.zeros(len(N_test))
    tykkelse_bunn_1 = tykkelse_kryssfiner + tykkelse_absorbent_bunn_20mm
    for i, val in enumerate(N_test):
        
        d_QRD_maks_1[i], b_QRD_1[i] = _parametere_QRD(d_box, tykkelse_bunn_1, b_box, tykkelse_kryssfiner, val)
        n_maks_1[i], f_0_QRD_1[i], n_1 = _elementnummer_dybde_og_f_0(val,d_QRD_maks_1[i])
        QRD_f_nedre_verifisering_1[i] = 344 / (val * b_QRD_1[i])
        QRD_f_nedre_1[i] = 0.5 * f_0_QRD_1[i]
        QRD_f_øvre_1[i] = 573 / b_QRD_1[i]


    n_1 = []
    n_cm_1 = []
    for m in range(N*2):
        n_1.append(m**2 % N)

    n_maks_1 = max(n_1)
    for i in n_1: n_cm_1.append(i*(d_QRD_maks_1[0] / n_maks_1))
    n_cm_1 = n_cm_1[(N+1)//2:N+N//2+2]

    x_steps_1 = np.linspace(tykkelse_kryssfiner, tykkelse_kryssfiner+((N)*b_QRD[N_it]), N+1)

    for i,val in enumerate(n_cm_1): n_cm_1[i] = d_QRD_maks_1[0] - val






    #### Plot parametere
    N_x_values = []
    N_y_values = []
    y_ticks = [0]
    y_ticks_label = ["Box"]
    x_ticks_third_octave = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
    x_ticks_third_octave_labels = ["100", "200", "500", "1k", "2k", "5k","10k","20k","40k"]

    print("######## Tykkelse aborbent: {0} mm".format(tykkelse_absorbent_bunn_15mm*100))
    print("f_nedre 0.5*f_0\t\t f_nedre 344/(N*b)\t\t f_øvre 573/b")
    for i in range(len(N_test)):
        print(round(QRD_f_nedre[i],1)," Hz\t\t",round(QRD_f_nedre_verifisering[i],1)," Hz\t\t\t",round(QRD_f_øvre[i],1)," Hz")
        N_x_values.append([QRD_f_nedre[i],QRD_f_øvre[i]])
        N_y_values.append([i+1,i+1])
        y_ticks.append(i+1)
        y_ticks_label.append("N={0}".format(N_test[i]))

    N_x_values_1 = []
    N_y_values_1 = []
    y_ticks_1 = [0]
    y_ticks_label_1 = ["Box"]
    x_ticks_third_octave_1 = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
    x_ticks_third_octave_labels_1 = ["100", "200", "500", "1k", "2k", "5k","10k","20k","40k"]

    print("######## Tykkelse aborbent: {0} mm".format(tykkelse_absorbent_bunn_20mm*100))
    print("f_nedre 0.5*f_0\t\t f_nedre 344/(N*b)\t\t f_øvre 573/b")
    for i in range(len(N_test)):
        print(round(QRD_f_nedre_1[i],1)," Hz\t\t",round(QRD_f_nedre_verifisering_1[i],1)," Hz\t\t\t",round(QRD_f_øvre_1[i],1)," Hz")
        N_x_values_1.append([QRD_f_nedre_1[i],QRD_f_øvre_1[i]])
        N_y_values_1.append([i+1,i+1])
        y_ticks_1.append(i+1)
        y_ticks_label_1.append("N={0}".format(N_test[i]))

    
    fig = plt.figure(figsize=(10,10))
    plt.style.use("ggplot")
    ax = fig.add_subplot(221)
    ax1 = fig.add_subplot(223)
    ax_2 = fig.add_subplot(222)
    ax1_2 = fig.add_subplot(224)
    
    #### Frequency plot
    for i in range(len(N_test)):
        ax.plot(N_x_values[i],N_y_values[i],label="N={0}".format(N_test[i]),linewidth=5)
    ax.plot([box_f_nedre,box_f_øvre],[0,0],label="Box",linewidth=5)
    ax.set_xticks(x_ticks_third_octave)
    ax.set_xticklabels(x_ticks_third_octave_labels)     
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_label)
       
    ax.grid(which="major", color="dimgray")
    ax.grid(which="minor", linestyle=":", color="dimgray")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title("Frekvens og skisse for \ntykkelse absorbent: {0} mm".format(tykkelse_absorbent_bunn_15mm*1000))
    ax.set_xscale("log")
    
    
    #### Diffusor template
    y_shift = 0.125 + tykkelse_bunn
    y_shift1 = tykkelse_kryssfiner
    y_shift_2 = 0.125 + tykkelse_bunn_1
    y_shift1_2 = tykkelse_kryssfiner
    # Plot the first plot, shifted by y_shift
    ax1.step(x_steps, [y+y_shift for y in n_cm], where="post")
    ax1.vlines([0, tykkelse_kryssfiner, tykkelse_kryssfiner+N*b_QRD[N_it], 2*tykkelse_kryssfiner+N*b_QRD[N_it]], y_shift, d_QRD_maks[0]+y_shift, colors="black")
    ax1.hlines([d_QRD_maks[0]+y_shift, d_QRD_maks[0]+y_shift], [0, tykkelse_kryssfiner+N*b_QRD[N_it]], [tykkelse_kryssfiner, 2*tykkelse_kryssfiner+N*b_QRD[N_it]], colors="black")
    ax1.vlines([0, 2*tykkelse_kryssfiner+N*b_QRD[N_it]], y_shift, -tykkelse_bunn+y_shift, colors="black")
    ax1.hlines([y_shift, -tykkelse_bunn+y_shift], 0, 2*tykkelse_kryssfiner+N*b_QRD[N_it], colors="black")
    
    ax1.vlines([0,tykkelse_kryssfiner,tykkelse_kryssfiner+N*b_QRD[N_it],2*tykkelse_kryssfiner+N*b_QRD[N_it]],y_shift1, d_box/2 - tykkelse_kryssfiner+y_shift1,colors="black")
    ax1.hlines([d_box/2 - tykkelse_kryssfiner+y_shift1,d_box/2 - tykkelse_kryssfiner+y_shift1],[0,tykkelse_kryssfiner+N*b_QRD[N_it]],[tykkelse_kryssfiner,2*tykkelse_kryssfiner+N*b_QRD[N_it]],colors="black")
    ax1.vlines([0, 2*tykkelse_kryssfiner+N*b_QRD[N_it]],y_shift1,-tykkelse_kryssfiner+y_shift1,colors="black")
    ax1.hlines([y_shift1,-tykkelse_kryssfiner+y_shift1],0,2*tykkelse_kryssfiner+N*b_QRD[N_it],colors="black")

    ax1.set_xlim(-b_box/4, b_box+b_box/4)
    ax1.set_ylim(-b_box/4, b_box+b_box/4)
    #ax1.set_title("Snitt av QRD del for diffusjonsbokser med N={0}".format(N))
    ax1.set_ylabel("Dybde [m]")
    ax1.set_xlabel("Bredde [m]")

    for i in range(len(N_test)):
        ax_2.plot(N_x_values_1[i],N_y_values_1[i],label="N={0}".format(N_test[i]),linewidth=5)
    ax_2.plot([box_f_nedre,box_f_øvre],[0,0],label="Box",linewidth=5)
    ax_2.set_xticks(x_ticks_third_octave_1)
    ax_2.set_xticklabels(x_ticks_third_octave_labels_1)     
    ax_2.set_yticks(y_ticks_1)
    ax_2.set_yticklabels(y_ticks_label_1)
    
    ax_2.grid(which="major", color="dimgray")
    ax_2.grid(which="minor", linestyle=":", color="dimgray")
    ax_2.set_xlabel("Frequency [Hz]")
    ax_2.set_title("Frekvens og skisse for \ntykkelse absorbent: {0} mm".format(tykkelse_absorbent_bunn_20mm*1000))
    ax_2.set_xscale("log")

    # Plot the first plot, shifted by y_shift_1
    ax1_2.step(x_steps, [y+y_shift_2 for y in n_cm_1], where="post")
    ax1_2.vlines([0, tykkelse_kryssfiner, tykkelse_kryssfiner+N*b_QRD[N_it], 2*tykkelse_kryssfiner+N*b_QRD[N_it]], y_shift_2, d_QRD_maks_1[0]+y_shift_2, colors="black")
    ax1_2.hlines([d_QRD_maks_1[0]+y_shift_2, d_QRD_maks_1[0]+y_shift_2], [0, tykkelse_kryssfiner+N*b_QRD[N_it]], [tykkelse_kryssfiner, 2*tykkelse_kryssfiner+N*b_QRD[N_it]], colors="black")
    ax1_2.vlines([0, 2*tykkelse_kryssfiner+N*b_QRD[N_it]], y_shift_2, - tykkelse_bunn_1+y_shift_2, colors="black")
    ax1_2.hlines([y_shift_2, -tykkelse_bunn_1+y_shift_2], 0, 2*tykkelse_kryssfiner+N*b_QRD[N_it], colors="black")
    
    ax1_2.vlines([0,tykkelse_kryssfiner,tykkelse_kryssfiner+N*b_QRD[N_it],2*tykkelse_kryssfiner+N*b_QRD[N_it]],y_shift1_2, d_box/2 - tykkelse_kryssfiner+y_shift1_2,colors="black")
    ax1_2.hlines([d_box/2 - tykkelse_kryssfiner+y_shift1_2,d_box/2 - tykkelse_kryssfiner+y_shift1_2],[0,tykkelse_kryssfiner+N*b_QRD[N_it]],[tykkelse_kryssfiner,2*tykkelse_kryssfiner+N*b_QRD[N_it]],colors="black")
    ax1_2.vlines([0, 2*tykkelse_kryssfiner+N*b_QRD[N_it]],y_shift1_2,-tykkelse_kryssfiner+y_shift1_2,colors="black")
    ax1_2.hlines([y_shift1_2,-tykkelse_kryssfiner+y_shift1_2],0,2*tykkelse_kryssfiner+N*b_QRD[N_it],colors="black")

    
    ax1_2.set_xlim(-b_box/4, b_box+b_box/4)

    ax1_2.set_ylim(-b_box/4, b_box+b_box/4)
    #ax1.set_title("Snitt av QRD del for diffusjonsbokser med N={0}".format(N))
    ax1_2.set_ylabel("Dybde [m]")
    ax1_2.set_xlabel("Bredde [m]")

    #plt.legend()
    plt.show()
    
    
    return 1


_print_frequency_with_QRD(d_box, b_box, tykkelse_absorbent_bunn_15mm)





##### Run Main code from here #####

if __name__ == "__main__":
    
    
    
    
    
    
    
    
    print("#################################################################\n")
    d, b, l = _parametere_frekvens(f_0Box, f_nedregrense, f_øvregrense)
    d_maks, b_QRD = _parametere_QRD(d, tykkelse_kryssfiner, b, tykkelse_kryssfiner, N)
    _elementnummer_dybde_og_f_0(N, d_maks)
    _QRD_depth_calc(N, 400)
    _QRD_width_check(N, b_QRD)
    _QRD_øvre_frekvens(b_QRD)
    _plot_Box_shape_template(b,d_maks,b_QRD,N)
    print("\n#################################################################")
   







