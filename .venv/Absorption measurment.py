import numpy as np
import matplotlib.pyplot as plt



####### Font values #######
SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
###########################
third_octave_center_frequencies = [100, 125, 160, 200, 250,
            315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000, 6300, 8000, 10000]


x_ticks_third_octave = [100, 200, 500, 1000, 2000, 5000]#, 10000]
x_ticks_third_octave_labels = ["100", "200", "500", "1k", "2k", "5k"]


poly_deg = 8
poly_deg_2 = 3
decay_Curve = 2   # Set to 0=EDT, 1=T10, 2=T20, 3=T30
title_str = ["EDT(Early Decay Time) values", "Reverberation time (T10)", "Reverberation time (T20)", "Reverberation time (T30)"]


## Regn ut reg. linje




















def replace_comma(value):
    return float(value.decode().replace(',', '.'))

path = "C:/Users/erlen/TFE4940-Master-thesis/.venv/ABS measurements"

ABS_Obj = []
ABS_Wall = []
REF = []

for i in range(1,7):
    
    m_ABS_Obj =np.genfromtxt("{0}/m_ABS_{1}.txt".format(path,i), skip_header=4,skip_footer=2, usecols=(3, 4, 5, 6), converters={3: replace_comma, 4: replace_comma, 5: replace_comma, 6: replace_comma})
    u_ABS = np.genfromtxt("{0}/u_ABS_{1}.txt".format(path,i), skip_header=4,skip_footer=2, usecols=(3, 4, 5, 6), converters={3: replace_comma, 4: replace_comma, 5: replace_comma, 6: replace_comma})
    m_ABS_Wall = np.genfromtxt("{0}/m_ABS_Wall_{1}.txt".format(path,i), skip_header=4,skip_footer=2, usecols=(3, 4, 5, 6), converters={3: replace_comma, 4: replace_comma, 5: replace_comma, 6: replace_comma})

    ABS_Obj.append(m_ABS_Obj[:,decay_Curve])
    ABS_Wall.append(m_ABS_Wall[:,decay_Curve])
    REF.append(u_ABS[:,decay_Curve])

avg_Obj = []
avg_Wall = []
avg_ref = []
std_Obj = []
std_Wall = []
std_ref = []

print(np.array(REF)[:,0])


for i in range(0,21):
    avg_Obj.append(np.round(np.mean(np.array(ABS_Obj)[:,i]),2))
    avg_Wall.append(np.round(np.mean(np.array(ABS_Wall)[:,i]),2))
    avg_ref.append(np.round(np.mean(np.array(REF)[:,i]),2))
    std_Obj.append(np.std(np.array(ABS_Obj)[:,i]))
    std_Wall.append(np.std(np.array(ABS_Wall)[:,i]))
    std_ref.append(np.std(np.array(REF)[:,i]))

    
avg_Obj = np.array(avg_Obj)
avg_Wall = np.array(avg_Wall)
avg_ref = np.array(avg_ref)
std_Obj = np.array(std_Obj)
std_Wall = np.array(std_Wall)
std_ref = np.array(std_ref)
    
print(avg_Obj)
print(avg_ref)
print(avg_Wall)
abs_Coeff_Wall = np.zeros((21))
abs_Coeff_Obj = np.zeros((21))





#style = "seaborn-v0_8-paper"
#style = "classic"
style = "bmh"
#style = "dark_background"
#style = "ggplot"
#style = "Solarize_Light2"
#style = "fast"



fig = plt.figure(figsize=(9,10))
plt.style.use(style)
ax = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
ax.semilogx(third_octave_center_frequencies, avg_Obj, label="Mean value - Object", marker="v")
ax.fill_between(third_octave_center_frequencies, avg_Obj - std_Obj, avg_Obj + std_Obj, alpha=0.3, label="STD - Object" )
ax.semilogx(third_octave_center_frequencies, avg_Wall, label="Mean value - Planse absorber", marker="o")
ax.fill_between(third_octave_center_frequencies, avg_Wall - std_Wall, avg_Wall + std_Wall, alpha=0.3, label="STD - Plane absorber")
ax.semilogx(third_octave_center_frequencies, avg_ref, label="Mean value - Reference", marker="s")
ax.fill_between(third_octave_center_frequencies, avg_ref - std_ref, avg_ref + std_ref, alpha=0.3, label="STD - Reference")
ax.legend()
ax.set_xticks(x_ticks_third_octave)
ax.set_xticklabels(x_ticks_third_octave_labels)


ax.grid(which="major", color="dimgray")
ax.grid(which="minor", linestyle=":", color="dimgray")
ax.set_ylabel("Time [s]")
ax.set_xlabel("Frequency [Hz]")
#ax.set_title(title_str[decay_Curve])


for i in range(0, 21):
    abs_Coeff_Wall[i] = ((55.3 * 8.43 * 6.04 * 5.18) / (343 * 0.27 * 1.77 * 14)) * (
                (1 / avg_Wall[i]) - (1 / avg_ref[i]))
    abs_Coeff_Obj[i] = ((55.3 * 8.43 * 6.04 * 5.18) / (343 * 7)) * ((1 / avg_Obj[i]) - (1 / avg_ref[i]))


ax1.semilogx(third_octave_center_frequencies, abs_Coeff_Obj, label="Absorption - object", linestyle="--")
ax1.semilogx(third_octave_center_frequencies, abs_Coeff_Wall, label="Absorption - Wall", linestyle="-.")

coeff = np.polyfit(third_octave_center_frequencies, (abs_Coeff_Obj+abs_Coeff_Wall)*0.5, poly_deg)
polynomial = np.poly1d(coeff)
y_line=polynomial(third_octave_center_frequencies)
coeff_2 = np.polyfit(third_octave_center_frequencies, (abs_Coeff_Obj+abs_Coeff_Wall)*0.5, poly_deg_2)
polynomial_2 = np.poly1d(coeff_2)
y_line_2=polynomial_2(third_octave_center_frequencies)


ax1.semilogx(third_octave_center_frequencies, y_line, label="Fitted line of degree: {0}".format(poly_deg), linewidth=2)
ax1.semilogx(third_octave_center_frequencies, y_line_2, label="Fitted line of degree: {0}".format(poly_deg_2), linewidth=2)

ax1.set_xticks(x_ticks_third_octave)
ax1.set_xticklabels(x_ticks_third_octave_labels)
ax1.grid(which="major", color="dimgray")
ax1.grid(which="minor", linestyle=":", color="dimgray")
ax1.set_ylabel("Absorption coefficient [$\\alpha_s$]")
ax1.set_xlabel("Frequency [Hz]")
#ax1.set_title("Absorption coefficient $\\alpha_s$")
ax1.legend()
ax1.set_ylim(0,1)
#ax1.set_xscale("log")
#fig.text(0.5, 0.04, 'Frequency [Hz]', ha='center')

plt.tight_layout()
#plt.show()

fig.savefig("RT and absorption curves - {0} - style {1}.png".format(title_str[decay_Curve], style))
plt.close(fig)
        