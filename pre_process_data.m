function [hq,pq]=pre_process_data(Nstates,Kquant,ktrain)
% This function reads the input files that store recorded sound "a" spoken
% by healthy people (HC, healthy control) and patients affected by 
% Parkinson's Disease (PD)
% Then the function finds the fundamental frequency (different for each
% person) and resamples the signal taking Nstates-1 equally spaced samples
% between two peaks of the signal
% Then the function uses Kmeans (Max Lloyd quantization) on the subset of
% training signals to find the quantization thresholds (Kquant quantized levels)
% and generates the new quantized signals
% Nstates= number of samples between two peaks
% Kquant= number of quantization levels
% ktrain= vector that stores the indexes of the files used for training
% hq= cell vector containing the quantized signals of healthy people
% pq= cell vector containing the quantized signals of people affected by PD
%..............

FS=8000;% sampling rate in the original files
NN=2*FS;% 2 seconds of speech selected in the center part of the recorded signal
Nout=200*Nstates;% samples stored in the files for subsequent processing
K=10;%number of healthy speakers = number of speakers affected by PD
listh={'H000a1','H001a1','H002a1','H003a1','H004a1','H005a1','H006a1','H007a1','H008a1','H009a1'};
listp={'P000a1','P001a1','P002a1','P003a1','P004a1','P005a1','P006a1','P007a1','P008a1','P009a1'};
% resampling
h=zeros(Nout,K);p=h;
for k=1:K
    FILENAME=['./data/healthy/',listh{k},'.wav']
    [h_in, FS]=audioread(FILENAME);
    h_res=gen_resampled_data(h_in,FS,Nstates,NN,Nout);
    h(:,k)=h_res;
    FILENAME=['./data/parkins/',listp{k},'.wav']
    [p_in, FS]=audioread(FILENAME);
    p_res=gen_resampled_data(p_in,FS,Nstates,NN,Nout);
    p(:,k)=p_res;
end
% quantization
% ----------------healthy control
% find the centroids for the training signals
Z=h(:,ktrain);[NR,NC]=size(Z);Z=reshape(Z,NR*NC,1);[~,C]=kmeans(Z,Kquant);C=sort(C);%C stores the centroids
% perform quantization on all the signals
hq=cell(1,K);% K 
for k=1:K
    hk=h(:,k)';hkM=kron(hk,ones(Kquant,1));cM=kron(ones(1,NR),C);b=abs(hkM-cM);[~,I]=min(b,[],1); 
    hq{k}=I;
end
% ----------------PD patients
% find the centroids for the training signals
Z=p(:,ktrain);[NR,NC]=size(Z);Z=reshape(Z,NR*NC,1);[~,C]=kmeans(Z,Kquant);C=sort(C);%C stores the centroids
% perform quantization on all the signals
pq=cell(1,K);
for k=1:K
    pk=p(:,k)';pkM=kron(pk,ones(Kquant,1));cM=kron(ones(1,NR),C);b=abs(pkM-cM);[~,I]=min(b,[],1); 
    pq{k}=I;
end

return
%%%% -----------------------
function a_out=gen_resampled_data(a_in,fs,Nstates,NN,Nout)
% This function resamples the input signal
%..............
% select the central part of the signal (NN samples)
Na=length(a_in);Nac=floor(Na/2);x=a_in(Nac-NN/2:Nac+NN/2-1);
% normalize the signal
x=x-mean(x);
s=sqrt(mean(x.^2));%root mean square
x=x/s;% normalized signal
% find the fundamental frequency
f=[0:NN-1]/NN*fs;
X=abs(fft(x));X=X(1:NN/2);% take only the first half samples of the FFT
Fmin=100;% at least 100 Hz between two adjacent peaks
[PKSa,LOCSa] = findpeaks(X,NN/fs,'MinPeakDistance',Fmin,'MinPeakProminence',200);
f0=LOCSa(1)%fundamental frequency
% find the peaks of x in the time domain at minimum distance 0.9/f0
t=[0:NN-1]/fs;
[PKSa,LOCSa] = findpeaks(x,fs,'MinPeakDistance',0.9/f0);
% take Nstates smples between two adjacent peaks
tt=zeros((length(LOCSa)-1)*Nstates,1);% new time axis
v=[0:Nstates-1]/Nstates;%evenly distributed Nstates values between 0 and 1
for jj=2:length(LOCSa)
    tt((jj-2)*Nstates+1:(jj-1)*Nstates)=v*(LOCSa(jj)-LOCSa(jj-1))+LOCSa(jj-1);
end
x1=interp1(t,x,tt);% resample signal x at the time instants defined by tt
% take only Nout samples
x1=x1(1:Nout);
% normalize again
a_out=x1-mean(x1);
a_out=a_out./std(a_out);
return
