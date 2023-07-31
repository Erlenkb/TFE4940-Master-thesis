
close all
clear 
clc
 
% PART ONE IMPEDANCE TUBE 
% All equations from ISO10534-2
% Input (section 7.2 and method document):
T   = 291;     % Assumed temperature
rho = 1.225;
c   = 343.2*sqrt(T/293);       % eq. (5)
 
x_1 = 0.180;       %Distance mic 1
d   = 0.1;         %Diameter of tube
s   = 0.08;        %Distance between microphones
 
%% Working frequency range 4.2
f_l =  100;          
f_u = .58*c/d;    
f_us = .45*c/s;
 
%% Load .etx files to Matlab
path2files = "C:\Users\erlen\TFE4940-Master-thesis\Matlab"
fileName1  = "100mm100-10000Hz_12.etx"
fileName2  = "100mm100-10000Hz_21.etx"
filename1_2 = "12mm100-10000Hz_12.etx"
filename2_2 = "12mm100-10000Hz_21.etx"
filename3  = "TestMaterial_Glava.etx"




Tube_f1    = fileName1;
Tube_f2    = fileName2;
Tube_f1_2  = filename1_2;
Tube_f2_2  = filename2_2;
Tube_f3    = filename3;


Psamp1     = importdata(Tube_f1,'\t',22); % 22 is the number of header lines
Psamp2     = importdata(Tube_f2,'\t',22);
Psamp1_2   = importdata(Tube_f1_2,'\t',22);
Psamp2_2   = importdata(Tube_f2_2,'\t',22);
Psamp3     = importdata(Tube_f3,'\t',22);


%% Extract data
tt      = Psamp1.data(:,1)';
ind     = 100e-3/(tt(2)-tt(1));          % Extract a part of the signal
 
tt      = tt(1:ind);
p1(1,:) = Psamp1.data(1:ind,3)'; % Mic 1 in coulmn 2 for rec. 1 - to p1
p1(2,:) = Psamp1.data(1:ind,2)';
p2(1,:) = Psamp2.data(1:ind,2)'; % Mic 1 in coulmn 2 for rec. 2 - to p1
p2(2,:) = Psamp2.data(1:ind,3)';                         

p1_2(1,:) = Psamp1_2.data(1:ind,3)'; % Mic 1 in coulmn 2 for rec. 1 - to p1
p1_2(2,:) = Psamp1_2.data(1:ind,2)';
p2_2(1,:) = Psamp2_2.data(1:ind,2)'; % Mic 1 in coulmn 2 for rec. 2 - to p1
p2_2(2,:) = Psamp2_2.data(1:ind,3)'; 

p3(1,:) = Psamp3.data(1:ind,3)'; % Mic 1 in coulmn 2 for rec. 1 - to p1
p3(2,:) = Psamp3.data(1:ind,2)';
p4(1,:) = Psamp3.data(1:ind,3)'; % Mic 1 in coulmn 2 for rec. 2 - to p1
p4(2,:) = Psamp3.data(1:ind,2)'; 



% Important to have control of these

%% Plot for quality control and prep question




figure(11)
subplot(2,2,1)
plot1=plot(tt,p1)
hold on
plot2=plot(tt,p2)
ylim([-0.2 0.4])
legend("p1^I", "p2^I", "p2^{II}","p1^{II}" ,"Location", "best")
hold off
grid on
xlabel('Time [s] '), ylabel('Magnitude'), title( 'Impulse response ');
%xlim([0, 0.008]),ylim([-3,2]), legend([p1(1,1)], 'p1', 'p2')
set(gca,'fontsize',12,'fontweight','bold');
set(gcf,'units','centimeters','position',[2,1,29.7,21.0]) % Set size of plot to A4 size
 
%% Start computation 
% FFT
n  = 2^nextpow2( size(p1,2) );                          % Number of element in signal into fft 
fs = 44100;                              % (should be 2 to some power for better execution of fft)        
df = fs / n;

ff = fs*(0:(n-1))/n;            % Frequency vector
ww = 2*pi*ff;                   % Angular frequency

frecvec1 = fft(p1(1,:) ,n);     % Repeat for all data
frecvec2 = fft(p1(2,:) ,n);
frecvec3 = fft(p2(2,:) ,n);     % Repeat for all data
frecvec4 = fft(p2(1,:) ,n);

frecvec1_2 = fft(p1_2(1,:) ,n);     % Repeat for all data
frecvec2_2 = fft(p1_2(2,:) ,n);
frecvec3_2 = fft(p2_2(2,:) ,n);     % Repeat for all data
frecvec4_2 = fft(p2_2(1,:) ,n);

frecvec1_3 = fft(p3(1,:) ,n);     % Repeat for all data
frecvec2_3 = fft(p3(2,:) ,n);
frecvec3_3 = fft(p4(2,:) ,n);     % Repeat for all data
frecvec4_3 = fft(p4(1,:) ,n);

%% Plot fft
figure(20) 

semilogx(ff,20*log10(frecvec1));
hold on;
semilogx(ff,20*log10(frecvec2));
semilogx(ff,20*log10(frecvec3));
semilogx(ff,20*log10(frecvec4));
title("FFT of P1");
xlim([f_l f_u]);
hold off;

%% Transfer function - see ISO standard for equations
%% Transfer function 100mm
H12I  = frecvec2 ./ frecvec1;           
H12II = frecvec3 ./ frecvec4;      

HC    = sqrt(H12I ./ H12II);    

H12   = H12I ./ HC;

HI    = exp(-1i*ww/c  *s );
HR    = exp( 1i*ww/c  *s );  

%% Transfer function 12mm
H12I_2  = frecvec2_2 ./ frecvec1_2;           
H12II_2 = frecvec3_2 ./ frecvec4_2;      

HC_2    = sqrt(H12I_2 ./ H12II_2);    

H12_2   = H12I_2 ./ HC_2;

HI_2    = exp(-1i*ww/c  *s );
HR_2    = exp( 1i*ww/c  *s ); 

%% Transfer function Glava 100mm
H12I_3  = frecvec2_3 ./ frecvec1_3;           
H12II_3 = frecvec3_3 ./ frecvec4_3;      

HC_3    = sqrt(H12I_3 ./ H12II_3);    

H12_3   = H12I_3 ./ HC_3;

HI_3    = exp(-1i*ww/c  *s );
HR_3    = exp( 1i*ww/c  *s );

 
%% Compute the absorption coefficient (alpha) and impedance (Z)
R =( (H12 - HI)./(HR - H12) ).*exp(2*1i*(ww/c)*x_1);    % eq. 17;
R_2=( (H12_2 - HI_2)./(HR_2 - H12_2) ).*exp(2*1i*(ww/c)*x_1);
R_3=( (H12_3 - HI_3)./(HR_3 - H12_3) ).*exp(2*1i*(ww/c)*x_1);



alpha = 1 - abs(R).^2;
alpha_2 = 1 - abs(R_2).^2;
alpha_3 = 1 - abs(R_3).^2;
Z = (1+R) ./ (1-R) * rho * c;
 
%% Plotting Absorbtion coefficients

figure(100)
semilogx(ff, alpha);
hold on
semilogx(ff, alpha_2);
semilogx(ff, alpha_3);
grid on
legend("100mm Himling", "12mm himling", "100mm Glava", "Location", "best");
title(" Absorption coefficient");
set(gca,"Fontsize",12,"fontweight","bold");
xlabel("Frequency [Hz]");
ylabel("Magnitude [Coefficient]");
xlim([f_l f_u]);
ylim([0 1]);
hold off

%% Plotting final results
 
figure(11)
subplot(2,2,2)
semilogx(ff,20*log10(frecvec1));
hold on;
semilogx(ff,20*log10(frecvec2));
semilogx(ff,20*log10(frecvec3));
semilogx(ff,20*log10(frecvec4));
xlabel("Frequency [Hz]");
ylabel("Magnitude");
title("FFT of the Impulse respons")
set(gca,'fontsize',12,'fontweight','bold');
legend("FFT(p1^I)","FFT(p2^I)","FFT(p2^{II})" ,"FFT(p1^{II})","Location", "best")
grid on
%ylim
xlim([f_l f_u]);
hold off
 
figure(11)
subplot(2,2,3)
semilogx(ff,alpha);
hold on
semilogx(ff,abs(R));
grid on
legend("Absorption", "Reflection", "Location", "best")
hold off
xlim([f_l f_u]);
ylim([0 1]);
xlabel('Frequency [Hz]'), ylabel('Magnitude')
title('Absorption-and Reflection coefficient ')
set(gca,'fontsize',12,'fontweight','bold');


figure(11)
subplot(2,2,4)
semilogx(ff,real(Z),'r')
hold on
semilogx(ff,imag(Z),'b');
semilogx(ff,abs(Z),'k');
grid on
% ylim([-4 4])
xlim([f_l f_u]);
title('Acoustic Impedance Z')
xlabel('Frequency [Hz]' )
legend('Re[Z]', "Im[Z]", "|Z|", "Location", "best")
set(gca,'fontsize',12,'fontweight','bold');
hold off
 
%% Mikis model

f = [100:1:2000];
omega = 2*pi*f;

rho_0 = 1.225;      % [Kg.m-3] density at rest of air at 18C, 1atm
c_0   = 342.2;      % [m.s-1] speed of sound in air at 18C, 1atm
P_0   = 1.0132e+05; % [N.m-2] atmospheric pressure at 18C, 1atm
sigma = 9100       % [N.s.m-4] static air flow resistivity of material
h     = 0.1        % [m] thickness of material
X = f/sigma;

Z_DB70_Mik90 = rho_0*c_0*( 1 + 5.50*(X*1000).^(-0.632) ...
                            - i*8.43*(X*1000).^(-0.632) ); 

k_DB70_Mik90 = omega/c_0 .* (-i) .* ( 11.41*(X*1000).^(-0.618) ...
                                      + i* (1 + 7.81*(X*1000).^(-0.618) ) );

Z = Z_DB70_Mik90;
Z = -1i.*Z_DB70_Mik90./cot(k_DB70_Mik90*h);

R = (Z-rho_0*c_0)./(Z+rho_0*c_0);
a = 1-abs(R).^2

figure(10)
subplot(1,2,2)
semilogx(f, real(Z))
hold on
semilogx(f, imag(Z))
semilogx(f, abs(Z))
grid on
title('Specific Impedance Z_C')
xlabel('Frequency [Hz]' )
ylabel("Magnitude")
legend('Re[Z]', "Im[Z]", "|Z|")
set(gca,'fontsize',12,'fontweight','bold');
set(gcf,'units','centimeters','position',[2,1,29.7,11.0])
hold off

figure(10)
subplot(1,2,1)
semilogx(f, abs(R))
hold on
semilogx(f, abs(a))
title('Absorption-and Reflection Coefficient')
xlabel('Frequency [Hz]')
ylabel("Magnitude")
legend('Absorption', "Reflection", "Location", "best")
set(gca,'fontsize',12,'fontweight','bold');
grid on
hold off

%xlim([f_l f_u]);
%% Save the results
%exportgraphics(figure(10), ['Mikis_Model.png'],'Resolution',450)
%exportgraphics(figure(11), ['Impedance_Tube.png'],'Resolution',450)
exportgraphics(figure(100), ['Absorption Coefficients.png'],'Resolution',450)

 

%  1+1   % Ez math?
 
 
 
 
 


