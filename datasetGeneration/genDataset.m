close all, clear

% Parámetros para la generación de señales
SIGNALS_POR_MODULACION = 17000;
SNR = -20:5:30;
LONGITUD_SIGNAL = 1e3;

% Parámetros comunes para las señales
Fs = 200e3;           % Frecuencia de muestreo en Hz
t = 0:1/Fs:LONGITUD_SIGNAL/Fs;        % Vector de tiempo de 1 segundo


% Información transmitida
N = 1500;                         % Número de bits
dataBits = randi([0 1], N, 1);   % Vector de bits aleatorios

bitrate = 50e3; %bits por segundo

duracion_total_s = length(dataBits)/bitrate;
duracionbit = 1/bitrate;
muestras_bit = ceil(Fs*duracionbit);





%% Modulación de Amplitud (AM)



% for i= 1:1:SIGNALS_POR_MODULACION
%     Fc = 3000 + (rand * (10000-3000));            % Frecuencia de la portadora en Hz
%     fm = 1e3+(rand*(5e3-1e3));             % Frecuencia de la señal moduladora en Hz
%     m = cos(2*pi*fm*t);  % Señal moduladora
%     Am = 0.5 + (rand * 0.5);              % Amplitud de la portadora
%     Ac = Am / 2;         % Coeficiente de modulación
%     sAM = (1 + Ac*m) .* cos(2*pi*Fc*t);
%     snr = SNR(randi(length(SNR)));
%     sAm = awgn(sAM,snr,"measured");
%     guardarEspectrograma(sAM,Fs,"am_"+snr+"_"+i+".png")
% end
% 
% %% Modulación de Frecuencia (FM)
% for i= 1:1:SIGNALS_POR_MODULACION
%     Fc = 300 + (rand * (3e3-300));
%     fm = 10+(rand*190);             % Frecuencia de la señal moduladora en Hz
%     m = cos(2*pi*fm*t);  % Señal moduladora
%     freqdev =  20 + (rand * (200e3 - 20));
%     sFM= fmmod( m, Fc , Fs , freqdev );
%     snr = SNR(randi(length(SNR)));
%     sFM = awgn(sFM,snr,"measured");
%     guardarEspectrograma(sFM,Fs,"fm_"+snr+"_"+i+".png")
% end
% 
% 
% %% Modulación de Fase (PM)
% for i= 1:1:SIGNALS_POR_MODULACION
% 
%     phasedev =pi/2;
%     Fc = 300 + (rand * (30e3-300));            % Frecuencia de la portadora en Hz
%     fm = 10+(rand*190);             % Frecuencia de la señal moduladora en Hz
%     snr = SNR(randi(length(SNR)));
%     m = cos(2*pi*fm*t);  % Señal moduladora
%     sPM = pmmod(m,Fc,Fs,phasedev);
%     sPM = awgn(sPM,snr,"measured");
%     guardarEspectrograma(sPM,Fs,"pm_"+snr+"_"+i+".png")
% end
% 


%% BPSK
% for i= 1:1:SIGNALS_POR_MODULACION
%     bpskSig = pskmod(dataBits,2);  % Crear un modulador BPSK
%     snr = SNR(randi(length(SNR)));
%     bpskSig = repelem(bpskSig,muestras_bit);
%     bpskSig = awgn(bpskSig,snr,"measured");
%     bpskSig = bpskSig(1:LONGITUD_SIGNAL);
%     guardarEspectrograma(bpskSig,Fs,"bpsk_"+snr+"_"+i+".png")
% end

%% QPSK
for i= 1:1:SIGNALS_POR_MODULACION
    qpskSig = pskmod(dataBits,4);  % Crear un modulador BPSK
    qpskSig = repelem(qpskSig,muestras_bit);
    qpskSig = qpskSig(1:LONGITUD_SIGNAL);
    snr = SNR(randi(length(SNR)));

    qpskSig = awgn(qpskSig,snr,"measured");
    guardarEspectrograma(qpskSig,Fs,"qpsk_"+snr+"_"+i+".png")
end
%% 16-QAM
M = 16;                                        % Orden de la modulación
for i= 1:1:SIGNALS_POR_MODULACION

    qam16Sig = qammod(dataBits,M,InputType="bit");    % Crear un modulador 16-QAM
    % Se repiten las muestras para ajustarlo a Fs
    qam16Sig = repelem(qam16Sig,muestras_bit);
    qam16Sig = qam16Sig(1:LONGITUD_SIGNAL);
    snr = SNR(randi(length(SNR)));

    qam16Sig = awgn(qam16Sig,snr,"measured");
    guardarEspectrograma(qam16Sig,Fs,"16qam_"+snr+"_"+i+".png")
end
%% 8-QAM
M =8;                                        % Orden de la modulación
for i= 1:1:SIGNALS_POR_MODULACION

    qam8Sig = qammod(dataBits,M,InputType="bit");    % Crear un modulador 16-QAM
    % Se repiten las muestras para ajustarlo a Fs
    qam8Sig = repelem(qam8Sig,muestras_bit);
    qam8Sig = qam8Sig(1:LONGITUD_SIGNAL);
    snr = SNR(randi(length(SNR)));
    qam8Sig = awgn(qam8Sig,snr,"measured");
    guardarEspectrograma(qam8Sig,Fs,"8qam_"+snr+"_"+i+".png")
end
%% 32-QAM
M = 32;                                        % Orden de la modulación
for i= 1:1:SIGNALS_POR_MODULACION

    qam32Sig = qammod(dataBits,M,InputType="bit");    % Crear un modulador 16-QAM
    % Se repiten las muestras para ajustarlo a Fs
    qam32Sig = repelem(qam32Sig,muestras_bit);
    qam32Sig = qam32Sig(1:LONGITUD_SIGNAL);
    snr = SNR(randi(length(SNR)));
    qam32Sig = awgn(qam32Sig,snr,"measured");
    guardarEspectrograma(qam32Sig,Fs,"16qam_"+snr+"_"+i+".png")
end
%% 64-QAM
M = 64;                                        % Orden de la modulación
for i= 1:1:SIGNALS_POR_MODULACION
    qam64Sig = qammod(dataBits,M,InputType="bit");     % Crear un modulador 64-QAM
    qam64Sig = repelem(qam64Sig,muestras_bit);
    qam64Sig = qam64Sig(1:LONGITUD_SIGNAL);
    snr = SNR(randi(length(SNR)));
    qam64Sig = awgn(qam64Sig,snr,"measured");
    guardarEspectrograma(qam64Sig,Fs,"64qam_"+snr+"_"+i+".png")
end


%% ASK modulation
A1=1;      % Amplitude of carrier signal for information 1
A2=0;       % Amplitude of carrier signal for information 0
bp = 1/bitrate;
t2=0:1/Fs:bp;
for i= 1:1:SIGNALS_POR_MODULACION    

    f=2e3 + (rand * (500e3 - 2e3));    % carrier frequency
    
    askSig=[];
    for j=1:1:length(dataBits)
        if dataBits(j)==1
            y=A1*cos(2*pi*f*t2);
        else
            y=A2*cos(2*pi*f*t2);
        end
        askSig=[askSig y];
    end
    askSig = askSig(1:LONGITUD_SIGNAL);
    snr = SNR(randi(length(SNR)));
    askSig = awgn(askSig,snr,"measured");
    guardarEspectrograma(askSig,Fs,"ask_"+snr+"_"+i+".png")

end

%% FSK modulation
for i= 1:1:SIGNALS_POR_MODULACION    
    M= 2;
    freq_sep = 20 + (rand * (10e3 - 20));
    nsamp = 2;
    fskSig = fskmod(dataBits,M,freq_sep,nsamp,Fs);
    fskSig = fskSig(1:LONGITUD_SIGNAL);
    snr = SNR(randi(length(SNR)));
    fskSig = awgn(fskSig,snr,"measured");
    guardarEspectrograma(fskSig,Fs,"fsk_"+snr+"_"+i+".png")
end
%% Ruido
for i= 1:1:SIGNALS_POR_MODULACION 
    ruido = rand(LONGITUD_SIGNAL,1);
    guardarEspectrograma(ruido,Fs,"noise_"+i+".png")
end

%% OFDM modulation
numSubcarriers = [64,128,252];
numSymbols = 10;
qamOrder = [4,32,64];
cpLength = 12;

% for i= 1:1:SIGNALS_POR_MODULACION 
%     order = qamOrder(randi(length(qamOrder)));
%     subcarriers = numSubcarriers(randi(length(numSubcarriers)));
%     signalOFDM = generarOFDM(subcarriers, numSymbols, order, cpLength);
%     signalOFDM = awgn(signalOFDM,snr,"measured");
%     guardarEspectrograma(signalOFDM,Fs,"OFDM_"+snr+"_"+i+".png")
% end