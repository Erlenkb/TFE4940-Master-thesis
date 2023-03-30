import matplotlib.pyplot as plt
import numpy as np


####### Parameterverdier ######
tykkelse_kryssfiner = 0.02
f_0Box = 280
f_nedregrense = 200
f_øvregrense = 10000
N = 17

##############################
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



print("#################################################################\n")
d, b, l = _parametere_frekvens(f_0Box, f_nedregrense, f_øvregrense)
d_maks, b_QRD = _parametere_QRD(d, tykkelse_kryssfiner, b, tykkelse_kryssfiner, N)
_elementnummer_dybde_og_f_0(N, d_maks)
_QRD_depth_calc(N, 400)
_QRD_width_check(N, b_QRD)
_QRD_øvre_frekvens(b_QRD)

print("\n#################################################################")



def _plot_Box_shape_template(b,d_maks,b_QRD,N,tykkelse_kant=tykkelse_kryssfiner, tykkelse_bunn=tykkelse_kryssfiner):
    n = []
    n_cm = []
    for m in range(N*2):
        n.append(m**2 % N)
    
    print(n_cm)
    n_maks = max(n)
    for i in n: n_cm.append(i*(d_maks / n_maks))
    n_cm = n_cm[(N+1)//2:N+N//2+2]
    print(n_cm)
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
    ax.set_xlim(-b_QRD,b+b_QRD)
    ax.set_ylim(-b_QRD, b+b_QRD)
    ax.set_title("Snitt av QRD del for diffusjonsbokser med N={0}".format(N))
    ax.set_ylabel("dybde [m]")
    ax.set_xlabel("Bredde [m]")
    
    
    
    #print(x_steps)
    #print(n)
    
    
    plt.show()
   
    
    
    #ax.hlines()
    
    
    return 1


_plot_Box_shape_template(b,d_maks,b_QRD,N)

print(8//2)

