close all
clear 
clc


file_meas_exp = "16_S01_R01_exp.etx";
file_meas_ply = "16_S01_R01_ply.etx";
file_simu_ply = "S16_R16_Ply_2.MAT";
file_simu_exp = "S16_R16_Exp.MAT";
time_end = 11e-3;
file_FR = "M0020_S01_R01.etx";

psamp_FR = importdata(file_FR, "\t",22);
psamp1_m = importdata(file_meas_exp, "\t",22);
psamp1_s = load(file_simu_exp);
psamp2_m = importdata(file_meas_ply, "\t",22);
psamp2_s = load(file_simu_ply);

time_FR = psamp_FR.data(:,1)';
ind_end_FR =  11e-3/(time1_m(2)-time1_m(1));
ind1_start_FR = 6e-3/(time1_m(2)-time1_m(1));
time_FR = time_FR(ind1_start_FR:ind_end_FR);
val_FR = FR.data(ind1_start_FR:ind_end_FR,2)';


time1_m = psamp1_m.data(:,1)';
ind_end1_m =  time_end/(time1_m(2)-time1_m(1));
ind1_start1_m = 7e-3/(time1_m(2)-time1_m(1));
time1_m = time1_m(ind1_start1_m:ind_end1_m);
val1_m = psamp1_m.data(ind1_start1_m:ind_end1_m,2)' ;%*(1/max(abs(psamp1_m.data(:,2))));

time2_m = psamp2_m.data(:,1)';
ind_end2_m =  time_end/(time2_m(2)-time2_m(1));
ind2_start2_m = 7e-3/(time2_m(2)-time2_m(1));
time2_m = time2_m(ind2_start2_m:ind_end2_m);
val2_m = psamp2_m.data(ind2_start2_m:ind_end2_m,2)' ;%*(1/max(abs(psamp2_m.data(:,2))));

val1_s = psamp1_s.h_A0_01_OMNI;
Tt = 1/48000;

t_end = Tt*max(size(val1_s));
time1_s = 0:Tt:t_end;
ind_end1_s   = time_end/(time1_s(2)-time1_s(1));
ind1_start1_s = 7e-3/(time1_s(2)-time1_s(1));

val2_s = psamp2_s.h_A0_01_OMNI;
Tt = 1/48000;

t_end = Tt*max(size(val2_s));
time2_s = 0:Tt:t_end;
ind_end2_s   = time_end/(time2_s(2)-time2_s(1));
ind2_start2_s = 7e-3/(time2_s(2)-time2_s(1));


val1_s = val1_s(ind1_start1_s:ind_end1_s) ;%* (1/max(val1_s));
time1_s = time1_s(ind1_start1_s:ind_end1_s);
val2_s = val2_s(ind2_start2_s:ind_end2_s) ;%* (1/max(val2_s));
time2_s = time2_s(ind2_start2_s:ind_end2_s);

%%
figure(7)
subplot(1,2,1)
plt1 = plot(time1_s, val1_s)
hold on
plt2 = plot(time1_m,val1_m)
plt3 = plot(time2_s, val2_s)
plt4 = plot(time2_m,val2_m)

grid on
legend("CATT-A: Exp.", "Measured: Exp.", "CATT-A: Ply.", "Measured: Ply.","Location", "best");
xlabel("Time [ms]"), ylabel("Magnitude [Pa]");
title("Impulse Response")
set(gca,'fontsize',12,'fontweight','bold');
set(gcf,'units','centimeters','position',[2,1,29.7,11.0])
hold off

figure(8)
subplot(1,2,1)
plt5 = plot(time_FR,val_FR)
grid on
xlabel("Time [ms]"), ylabel("Magnitude [Pa]");
title("Impulse Response")
set(gca,'fontsize',12,'fontweight','bold');
set(gcf,'units','centimeters','position',[2,1,29.7,11.0])

n1_s  = 2^nextpow2( size(val1_s,2) );
fs1_s = 44100;
df1_s = fs1_s / n1_s;
n1_m = 2^nextpow2( size(val1_m,2) );
fs1_m = 48000;        
df1_m = fs1_m / n1_m;

n_FR = 2^nextpow2( size(val_FR,2) );
fs_FR = 48000;
ff_FR = fs_FR*(0:(n_FR-1))/n_FR;

ff1_s = fs1_s*(0:(n1_s-1))/n1_s;            % Frequency vector
ww1_s = 2*pi*ff1_s;                   % Angular frequency
ff1_m = fs1_m*(0:(n1_m-1))/n1_m;            % Frequency vector
ww1_m = 2*pi*ff1_m;

frecvec1_s = fft(val1_s ,n1_s);
frecvec1_m = fft(val1_m,n1_m); % - 23;
frecvec_FR = fft(val_FR, n_FR);

n2_s  = 2^nextpow2( size(val2_s,2) );
fs2_s = 44100;
df2_s = fs2_s / n2_s;
n2_m = 2^nextpow2( size(val2_m,2) );
fs2_m = 48000;        
df2_m = fs2_m / n2_m;

ff2_s = fs2_s*(0:(n2_s-1))/n2_s;            % Frequency vector
ww2_s = 2*pi*ff2_s;                   % Angular frequency
ff2_m = fs2_m*(0:(n2_m-1))/n2_m;            % Frequency vector
ww2_m = 2*pi*ff2_m;

frecvec2_s = fft(val2_s ,n2_s);
frecvec2_m = fft(val2_m,n2_m); % - 23;
%% 
figure(7) 
subplot(1,2,2)
semilogx(ff1_s,20*log10(frecvec1_s));
hold on
semilogx(ff1_m, 20*log10(frecvec1_m));
semilogx(ff2_s,20*log10(frecvec2_s));
semilogx(ff2_m, 20*log10(frecvec2_m));

xlim([100 22500]);
grid on
legend("CATT-A: Exp.", "Measured: Exp.", "CATT-A: Ply.", "Measured: Ply.","Location", "best");
xlabel("Frequency [Hz]"), ylabel("Magnitude [dB]");
title("Frequency Response");
set(gca,'fontsize',12,'fontweight','bold');
set(gcf,'units','centimeters','position',[2,1,29.7,11.0])
hold off

figure(8)
subplot(1,2,2)
semilogx(ff_FR, 20*log10(frecvec_FR));
xlim([100 22500]);
grid on

xlabel("Frequency [Hz]"), ylabel("Magnitude [dB]");
title("Frequency Response");
set(gca,'fontsize',12,'fontweight','bold');
set(gcf,'units','centimeters','position',[2,1,29.7,11.0])


%%
exportgraphics(figure(7), ['IR&FR_SR0_new.png'],'Resolution',450)
exportgraphics(figure(8), ['IR&FR_Tubespeaker.png'],'Resolution',450)