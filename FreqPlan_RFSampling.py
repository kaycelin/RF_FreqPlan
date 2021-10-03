name = "Grant Lin" 

import math as mh
import numpy as np
import scipy as signal
import matplotlib.pyplot as plt

def PLOT_FFT_dB(signalIn, fs, Nsamps=None, plt_format='plot', axis_scale_xy=[1,1], color=None, plt_legend=None, fnum=None):

    if type(signalIn) == np.ndarray and len(np.shape(signalIn)) == 2:
        N_plts = np.shape(signalIn)[0]
    elif type(signalIn) == np.ndarray and len(np.shape(signalIn)) == 1:
        N_plts = len(np.shape(signalIn))
        signalIn = signalIn.reshape(1,np.max(np.shape(signalIn)))

    if Nsamps is None:
        Nsamps = np.shape(signalIn)[1]
    freqs = np.arange(-Nsamps/2,Nsamps/2)*int(fs/Nsamps)

    # FFT
    signal_FFT = np.fft.fft(signalIn, Nsamps, axis=1)/1
    psd = np.fft.fftshift(np.abs(signal_FFT)/Nsamps, axes=1)**2
    
    if fnum is not None:
    #     return psd, freqs, signal_FFT 
    # else:
        ipwrdB_PSD_NoiseFloor = -150
        ipwr_PSD_NoiseFloor = 1e-3*10**(ipwrdB_PSD_NoiseFloor/10)
        psd_dB = 10*np.log10(np.around(np.abs(psd),30)+ipwr_PSD_NoiseFloor)

        # Plot
        if type(fnum) == int:
            plt.figure(fnum)
        elif type(fnum) == list:
            plt.figure(fnum[0])
            if len(fnum)==4:
                plt.subplot(fnum[1],fnum[2],fnum[3])
        elif type(fnum) == np.ndarray:
            plt.figure(fnum[0])
            if np.size(fnum)==4:
                plt.subplot(fnum[1],fnum[2],fnum[3])

        if len(axis_scale_xy)==1:
            axis_scale_xy = [axis_scale_xy[0],1]

        freqs_plt = freqs/axis_scale_xy[0]
        psd_dB_plt = psd_dB/axis_scale_xy[1]

        # plt2 = plt.twinx()
        for k in range(N_plts):
            if type(plt_legend) == list and len(plt_legend)==N_plts: # plt_legend=['HD1', 'HD2', 'HD3']
                plt_legend_k = plt_legend[k]
            elif type(plt_legend) == list and len(plt_legend)==1: # plt_legend=['HD1']
                plt_legend_k = plt_legend[0]+str(k+1)
            elif type(plt_legend) == str: # plt_legend='HD1'
                # plt_legend_k = plt_legend+str(k+1)
                plt_legend_k = plt_legend

            else: # plt_legend=None
                plt_legend_k = None

            if plt_format=='plot':
                # plt.plot(freqs_plt*1,psd_dB_plt[k],label=plt_legend_k)
                # psd_dB_plt[k,-1] = 0
                plt.fill_between(freqs_plt*1,ipwrdB_PSD_NoiseFloor,psd_dB_plt[k],label=plt_legend_k,alpha=0.8)
            elif plt_format=='scatter':
                plt.scatter(freqs_plt*1,psd_dB_plt[k],s=0.8, c=color,alpha=1,label=plt_legend_k)
            elif plt_format=='plot_R':
                plt2 = plt.twinx()
                plt2.plot(freqs_plt*1,psd_dB_plt[k],label=plt_legend_k)
                # plt2.fill(freqs_plt*1,psd_dB_plt[k],ipwrdB_PSD_NoiseFloor,label=plt_legend_k,alpha=0.8)
            elif plt_format=='scatter_R':
                plt2 = plt.twinx()
                plt2.scatter(freqs_plt*1,psd_dB_plt[k],s=0.8, c=color,alpha=1,label=plt_legend_k)
            
            if plt_legend_k != None and (plt_format == 'plot' or plt_format == 'scatter'):
                plt.legend(loc='upper right')
            elif plt_legend_k != None and plt_format == 'plot_R':
                plt.legend(loc='upper left')


    return signal_FFT, psd, freqs

def PLOT_FFT_Info(xRange=None,xStep=None,yRange=None,yStep=None,\
    xTicksLabel=None,yTicksLabel=None,xLim=None,yLim=None,\
        xLabel=None,yLabel=None,pltLegend=None,pltTitle=None,\
            axis_scale_xy=[1,1]):

    # ticks
    if xRange != None:
        plt_xticks = np.arange(xRange[0],xRange[1], step=(xStep))/axis_scale_xy[0]
        plt.xticks(plt_xticks,xTicksLabel)

    if  yRange != None and yStep == None:
        plt_yticks = yRange
        plt.yticks(plt_yticks,yTicksLabel)
    elif yRange != None and yStep != None:
        plt_yticks = np.arange(yRange[0],yRange[1], step=(yStep))/axis_scale_xy[1]
        plt.yticks(plt_yticks,yTicksLabel)

    # lim
    if xLim != None:
        plt.xlim(xLim[0],xLim[1])
    if yLim != None:
        plt.ylim(yLim[0],yLim[1])

    # axis label
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    if pltLegend != None:
        plt.legend(pltLegend)
    
    plt.title(pltTitle)
    # plt.show()


def mtones_HD_gen(fs, fbwInband_Fund, ipwrdB_Fund, Norder_HD, df=1, fnum=None):
    import numpy as np
    import matplotlib.pyplot as plt

    scale_fs = (np.fix(np.max(fbwInband_Fund*np.max(Norder_HD))/(fs/2))+1)
    fs_order = scale_fs*fs
    Nsamps = int(np.fix(fs_order/df))
    freqs = np.arange(-Nsamps/2,Nsamps/2)*df
    if type(Norder_HD) == int:
        N_HD = int(Norder_HD)
        n_HD = np.arange(1,Norder_HD+1).reshape(N_HD,1)

    elif type(Norder_HD) == np.ndarray:
        N_HD = np.size(Norder_HD)
        n_HD = Norder_HD.reshape(N_HD,1)

    Mtones = np.zeros((N_HD,Nsamps), dtype=complex)

    # NHD  
    freqs_N = np.tile(freqs,(N_HD,1))
    n_HD = np.arange(1,Norder_HD+1).reshape(N_HD,1)
    fbwInband_N = np.dot(n_HD,fbwInband_Fund.reshape(1,2))
    bwInand_BB_N = np.diff(fbwInband_N, axis=1)*np.array([-1,1])/2
    x1=np.abs(freqs_N-np.min(bwInand_BB_N,axis=1).reshape(N_HD,1))
    x2=np.min(np.abs(freqs_N-np.min(bwInand_BB_N,axis=1).reshape(N_HD,1)),axis=1).reshape(N_HD,1)
    ind_fL_N = np.where(x1==x2)[1].reshape(N_HD,1)
    x3=np.abs(freqs_N-np.max(bwInand_BB_N,axis=1).reshape(N_HD,1))
    x4=np.min(np.abs(freqs_N-np.max(bwInand_BB_N,axis=1).reshape(N_HD,1)),axis=1).reshape(N_HD,1)
    ind_fH_N = np.where(x3==x4)[1].reshape(N_HD,1)

    # Multitones
    plt_label='HD'
    plt_legend = []
    for k in range(N_HD):
        ind_inband = np.arange(ind_fL_N[k],ind_fH_N[k]+1)
        Mtones[k,ind_inband] = Nsamps*10**(ipwrdB_Fund[k]/20)*1
        x=plt_label+str(k+1)
        plt_legend.append(x)
    mtones = np.fft.ifft(np.fft.fftshift(Mtones, axes=1), axis=1)
    # PLOT_FFT_dB(mtones, fs_order, df, plt_format='plot', axis_scale_xy=list([1e6/scale_freq,1]), color=None, fnum=[101,2,1,1])

    # Upconversion
    fLO_N = (np.ceil(np.mean(fbwInband_N,axis=1)/df)*df).reshape(N_HD,1)
    t_N = np.tile(np.arange(0,Nsamps)/fs_order,(N_HD,1))
    loI_N = 1*np.cos(2*np.pi*fLO_N*t_N-0)*np.sqrt(2)*10**(1/20)
    loQ_N = 1*np.sin(2*np.pi*fLO_N*t_N+0)*np.sqrt(2)*10**(1/20)
    RMS_lo = np.mean([np.sqrt(np.mean(np.abs(np.square(loI_N[0])))),np.sqrt(np.mean(np.abs(np.square(loQ_N[0]))))])
    scale_real = np.sqrt(2)
    mtones_IQ_HDN = (np.real(mtones)*loI_N/RMS_lo - np.imag(mtones)*loQ_N/RMS_lo)*scale_real
    # PLOT_FFT_dB(mtones_IQ_HDN, fs_order, df, plt_format='plot', axis_scale_xy=list([1e6/scale_freq,1]), color='g', plt_legend=['HD'], fnum=[101,4,1,1])
    # PLOT_FFT_dB(mtones_IQ_HDN, fs_order, df, plt_format='plot', axis_scale_xy=list([1e6/scale_freq,1]), color='g', plt_legend='HD', fnum=[101,4,1,2])
    # PLOT_FFT_dB(mtones_IQ_HDN, fs_order, df, plt_format='plot', axis_scale_xy=list([1e6/1,1]), color='g', plt_legend=plt_legend, fnum=[101,4,1,3])
    # # PLOT_FFT_dB(mtones_IQ_HDN, fs_order, df, plt_format='plot', axis_scale_xy=list([1e6/scale_freq,1]), color='g', plt_legend=None, fnum=[101,4,1,4])

    return mtones_IQ_HDN, fs_order

def FP_RFSampling_UpDownSampling_wiFIR_g(signalIn, fsIn, fsOut):

    if type(signalIn) == np.ndarray and len(np.shape(signalIn)) == 2:
        N_HD = np.shape(signalIn)[0]
        Nsamps = np.shape(signalIn)[1]
    elif type(signalIn) == np.ndarray and len(np.shape(signalIn)) == 1:
        N_HD = len(np.shape(signalIn))
        Nsamps = np.shape(signalIn)[0]
        signalIn = signalIn.reshape(1,Nsamps)

    ratio_M = fsOut/fsIn
    if ratio_M <= 1:
        # Decimation
        signalSamp = signalIn[:,::int(1/ratio_M)]
        NsampsSamp = Nsamps*ratio_M
        # PLOT_FFT_dB(signalOut, fsOut, df=fsOut/NsampsOut, plt_format='plot', axis_scale_xy=[1e6,1], color=None, plt_legend=None, fnum=101)
        # plt.show()

    return signalSamp

def FIR_REMEZ(fbands,RippledB,fs,type='LPF'):
    if len(fbands) == 4 and type == 'BPF':
        fstop1 = fbands[0]
        fpass1 = fbands[1]
        fpass2 = fbands[2]
        fstop2 = fbands[3]
    elif len(fbands) == 2 and type == 'LPF':
        fpass1 = fbands[0]
        fstop1 = fbands[1]
        fpass2 = 0
        fstop2 = mh.inf
    else:
        print('error!')
        b = None
        Norder = None
        return b, Norder

    delta_ft1 = abs(fpass1-fstop1)/fs # freq. transition band1
    delta_ft2 = abs(fstop2-fpass2)/fs # freq. transition band2
    delat_ft_min = min(delta_ft1, delta_ft2)

    RPdB = RippledB[0]
    RSdB = RippledB[1]
    deltaP = 1-10**(-RPdB/20)
    deltaS = 10**(-RSdB/20)

    if type == 'BPF':
        a1 = 0.01201
        a2 = 0.09664
        a3 = -0.51325
        a4 = 0.00203
        a5 = -0.5705
        a6 = -0.44314
        Cinf = np.log10(deltaS) * (a1*np.log10(deltaP)**2 + a2*np.log10(deltaP) + a3) + (a4*np.log10(deltaP)**2 + a5*np.log10(deltaP) + a6)
        G = -14.6*np.log10(deltaS/deltaP) - 16.9
        Norder = int(Cinf/delat_ft_min + G*delat_ft_min + 1)
        ff = 1*np.array([0, fstop1, fpass1, fpass2, fstop2, fs/2])/(fs/2)
        aa = np.array([0, 0, 1, 1, 0, 0])
        wts_att = np.array([deltaP/deltaS, 1, deltaP/deltaS])
    elif type == 'LPF':
        a1 = 0.005309
        a2 = 0.07114
        a3 = -0.4761
        a4 = 0.00266
        a5 = 0.5941
        a6 = 0.4278
        Dinf = np.log10(deltaS) * (a1*np.log10(deltaP)**2 + a2*np.log10(deltaP) + a3) + (a4*np.log10(deltaP)**2 + a5*np.log10(deltaP) + a6)
        F = 11.01217 + 0.51244*(np.log10(deltaS) - np.log10(deltaP))
        Norder = int(Dinf/delat_ft_min + F*delat_ft_min + 1)
        ff = 1*np.array([0, fpass1, fstop1, fs/2])/(fs/2)
        aa = np.array([1, 1, 0, 0])
        wts_att = np.array([1.0, deltaP/deltaS])

    b = signal.remez(Norder, ff, aa[0::2], wts_att,Hz=2)
    numtaps, beta = signal.kaiserord(65, 24/(0.5*1000))

    return b, Norder

def FP_RFSampling_NCO_g(signalIn, fsIN, fc=None, fNCO=None):
    if type(signalIn) == np.ndarray and len(np.shape(signalIn)) == 2:
        N_HD = np.shape(signalIn)[0]
        Nsamps = np.shape(signalIn)[1]
    elif type(signalIn) == np.ndarray and len(np.shape(signalIn)) == 1:
        N_HD = len(np.shape(signalIn))
        Nsamps = np.shape(signalIn)[0]
        signalIn = signalIn.reshape(1,Nsamps)
    
    # fNCO = 660.88e6
    if fc == None and fNCO == None:
        print('Input error!')
        signalNCO = None
        fNCO = None
    elif fc != None and fNCO == None:
        # if np.mod(np.ceil(fc/(fsIN/2)),2) != 0:
        #     fNCO = fc-np.floor(fc/(fsIN/2))*fsIN/2
        # else:
        #     fNCO = np.ceil(fc/(fsIN/2))*fsIN/2-fc
        if fc > fs_ADC1:
            fNCO = -np.mod(fc, fs_ADC1)
        elif fc < fs_ADC1:
            fNCO = -np.mod(fs_ADC1, fc)

    fNCO_N = np.tile(fNCO,(N_HD,1))
    t_N = np.tile(np.arange(0,Nsamps)/fsIN,(N_HD,1))
    signalNCO = signalIn*np.exp(1J*2*np.pi*fNCO_N[0]*t_N[0])

    return signalNCO, fNCO

def ADC_Interleaving_Error_g(signalIn, M, Offset_LevelV=None, Offset_GaindB=None, Offset_PhaseDeg=None):

    Nsamps = np.shape(signalIn)[1]
    Ncarriers = np.shape(signalIn)[0]
    signalOut = np.zeros((Ncarriers,Nsamps),dtype=complex)

    for k in range(Ncarriers):
        signalIn_k = signalIn[k,:]
        # M = 4
        signalIn_kM = np.tile(signalIn_k,(M,1))
        signal_Intleav_kM = np.zeros((M,Nsamps),dtype=complex)

        if Offset_LevelV == None:
            Offset_Level = np.zeros((M,1))
        elif type(Offset_LevelV)==list and len(Offset_LevelV)==M:
            Offset_Level = np.array([Offset_LevelV]).T+0

        if Offset_GaindB == None:
            Offset_Gain = 10**(np.zeros((M,1))/10)
        elif type(Offset_GaindB)==list and len(Offset_GaindB)==M:
            Offset_Gain = 10**(np.array([Offset_GaindB]).T/10)

        if Offset_PhaseDeg == None:
            Offset_Phase = np.exp(1J*np.zeros((M,1)))
        elif type(Offset_PhaseDeg)==list and len(Offset_PhaseDeg)==M:
            Offset_Phase = np.exp(1J*np.array([Offset_PhaseDeg]).T/180*np.pi)

        signalOut_Offset_kM = (Offset_Level+signalIn_kM)*Offset_Gain*Offset_Phase
        # Inteleaving
        # x_interleaving(0,0:M:end) = signalOut_Offset_kM[0,0:M:end];
        # signal_Intleav_kM[0,0:-1:M]=signalOut_Offset_kM[0,0:-1:M]
        # signal_Intleav_kM[1,1:-1:M]=signalOut_Offset_kM[1,1:-1:M]
        # signal_Intleav_kM[2,2:-1:M]=signalOut_Offset_kM[2,3:-1:M]
        # signal_Intleav_kM[3,3:-1:M]=signalOut_Offset_kM[3,2:-1:M]

        signal_Intleav_kM[0,0::M]=signalOut_Offset_kM[0,0::M]
        signal_Intleav_kM[1,1::M]=signalOut_Offset_kM[1,1::M]
        signal_Intleav_kM[2,2::M]=signalOut_Offset_kM[2,2::M]
        signal_Intleav_kM[3,3::M]=signalOut_Offset_kM[3,3::M]

        signalOut[k,:] = np.sum(signal_Intleav_kM,axis=0)

        # a = np.tile(np.array([1,2,3,4,5,6,7,8,9,10,11,12]),(M,1))
        # Nsampsa = np.shape(a)[1]
        # b = np.zeros((M,Nsampsa),dtype=int)
        # b[0,0::M]=a[0,0::M]
        # b[1,1::M]=a[1,1::M]
        # b[2,2::M]=a[2,2::M]
        # b[3,3::M]=a[3,3::M]
        # a[0,i:i + n]
        # a[0,3::M]
        # a[0,1::M]


    return signalOut

# =============================================================================
import math as mh
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 

# Input, HD
# fbwInband_Fund = np.array([3.42e9,3.8e9])
fbwInband_Fund = np.array([3.475e9,3.525e9])
fc = np.mean(fbwInband_Fund)
Norder_HD = 3
ipwrdB_PSD_step = 30
ipwrdB_PSD_HD = (np.arange(0,Norder_HD)+1)*ipwrdB_PSD_step
# ipwrdB_PSD_NoiseFloor = -100

plt_ylim_delta = np.mean(np.diff(ipwrdB_PSD_HD))
plt_yRange_HD = list(np.append([0],ipwrdB_PSD_HD))
plt_ylim_HD = [np.min(plt_yRange_HD)-plt_ylim_delta,np.max(plt_yRange_HD)+plt_ylim_delta]
plt_yTicksLabel_HD = ['0','HD1','HD2','HD3']

# Input, RFBPF
flag_RFBPF = 'on'
# flag_RFBPF = 'off'
fbands_transition_RFBPF = 100e6
fbands_RFBPF = [fbwInband_Fund[0]-fbands_transition_RFBPF,fbwInband_Fund[0],fbwInband_Fund[1],fbwInband_Fund[1]+fbands_transition_RFBPF]*1
RippledB_RFBPF = [1,80]
if flag_RFBPF != 'on':
    RippledB_RFBPF = [0,0]

plt_yRange_RFBPF=[0, ipwrdB_PSD_HD[0]-RippledB_RFBPF[0]]
plt_yRange_RFBPF.extend(list(ipwrdB_PSD_HD[1:]-RippledB_RFBPF[1]))
plt_ylim_RFBPF = [np.min(plt_yRange_RFBPF)-plt_ylim_delta, np.max(plt_yRange_RFBPF)+plt_ylim_delta]

# Input, ADC Sampling and NCO
fs_ADC1 = 2949.12e6
# fs_ADC1 = 491.52e6*8
# fNCO = -660880000.0
if fc > fs_ADC1:
    fNCO = -np.mod(fc, fs_ADC1)
elif fc < fs_ADC1:
    fNCO = -np.mod(fs_ADC1, fc)
fs_ADC2 = 491.52e6

# Input, ADC1 Interleaving Offset
flag_ADCInterleav = 'on'
# flag_ADCInterleav = 'off'
M_Interleav = 4
Offset_LevelV=None
Offset_GaindB=None
Offset_PhaseDeg=None
Offset_LevelV=list(1*0.1*np.array([1, 3, -2, 0]))
Offset_GaindB=list(1*0.1*np.array([0.5,0,-0.5,0]))
Offset_PhaseDeg=list(1*0.1*np.array([3, 1, -1, -2]))

# Input, BBLPF
flag_BBLPF = 'on'
# flag_BBLPF = 'off'
fbands_transition_BBLPF = 20e6
fbands_BBLPF = [fs_ADC2/2,fs_ADC2/2+fbands_transition_BBLPF]*1
RippledB_BBLPF = [1,20]

plt_yRange_BBLPF = [0, plt_yRange_RFBPF[1]-RippledB_BBLPF[0]]
plt_yRange_BBLPF.extend(list(np.array(plt_yRange_RFBPF[2:])-RippledB_BBLPF[1]))
plt_ylim_BBLPF = [np.min(plt_yRange_BBLPF)-plt_ylim_delta, np.max(plt_yRange_BBLPF)+plt_ylim_delta]

# Setting, 
# df = 8e4
df = np.gcd(np.gcd(np.gcd(int(fbwInband_Fund[0]),int(fbwInband_Fund[1])),int(fs_ADC1)),int(fNCO))

flag_ON = [flag_RFBPF,flag_BBLPF,flag_ADCInterleav]
flag_ON.count('on')
fno = [101, 4+flag_ON.count('on'), 1, 1]
# fno = [101]

# =============================================================================

# generate HD
HDN, fs_order = mtones_HD_gen(fs_ADC1, fbwInband_Fund, ipwrdB_PSD_HD, Norder_HD, df)

HDN_FFT = PLOT_FFT_dB(HDN, fs_order, Nsamps=None, plt_format='plot', axis_scale_xy=[1e6,1], color=None, plt_legend=['HD'], fnum=fno)

PLOT_FFT_Info(xRange=[-fs_order/2,fs_order/2],xStep=fs_ADC1/2,yRange=plt_yRange_HD,yStep=None,\
    xTicksLabel=None,yTicksLabel=plt_yTicksLabel_HD,\
        xLim=None,yLim=plt_ylim_HD,\
            xLabel='Frequency(MHz)',yLabel=None,pltLegend=None,pltTitle='NHD',\
        axis_scale_xy=[1e6,1])

fno[-1]+=1

# generate RFBPF 
Nsamps=np.max(np.shape(HDN))
if flag_RFBPF == 'on':
    bRF,Ntaps_bRF = FIR_REMEZ(fbands_RFBPF,RippledB_RFBPF,fs_order,'BPF')
    fno[-1]-=1
    bRF_FFT = PLOT_FFT_dB(bRF*Nsamps, fs_order, Nsamps=Nsamps, plt_format='plot_R', axis_scale_xy=[1e6,1], color=None, plt_legend='RF BPF', fnum=fno)
    fno[-1]+=1

    # apply filter by freq. domain
    HDN_bRF = np.fft.ifft((bRF_FFT[0]/Nsamps*HDN_FFT[0]), axis=1)
    # apply filter by time. domain
    # HDN_bRF = np.zeros(np.shape(HDN))
    # for k in range(np.shape(HDN)[0]):
    #     HDN_bRF[k] = np.convolve(HDN[k], bRF, mode='same')
    PLOT_FFT_dB(HDN_bRF, fs_order, Nsamps=None, plt_format='plot', axis_scale_xy=[1e6,1], color=None, plt_legend=['HD+RFBPF_'], fnum=fno)
    
    PLOT_FFT_Info(xRange=[-fs_order/2,fs_order/2],xStep=fs_ADC1/2,yRange=plt_yRange_RFBPF,yStep=None,\
        xTicksLabel=None,yTicksLabel=plt_yTicksLabel_HD,\
            xLim=None,yLim=plt_ylim_RFBPF,\
                xLabel='Frequency(MHz)',yLabel=None,pltLegend=None,pltTitle='NHD+RFBPF',\
            axis_scale_xy=[1e6,1])

    fno[-1]+=1
else:
    HDN_bRF = HDN

# apply 1st sampling by ADC1
HDN_ADC1 = FP_RFSampling_UpDownSampling_wiFIR_g(HDN_bRF, fs_order, fs_ADC1)

PLOT_FFT_dB(HDN_ADC1, fs_ADC1, Nsamps=None, plt_format='plot', axis_scale_xy=[1e6,1], color=None, plt_legend=['HDN_ADC1_'], fnum=fno)

PLOT_FFT_Info(xRange=[-fs_ADC1/2,fs_ADC1/2],xStep=fs_ADC1/2,yRange=plt_yRange_RFBPF,yStep=None,\
    xTicksLabel=None,yTicksLabel=plt_yTicksLabel_HD,\
        xLim=None,yLim=plt_ylim_RFBPF,\
            xLabel='Frequency(MHz)',yLabel=None,pltLegend=None,pltTitle='RFSampling: '+str(fs_ADC1/1e6)+'MHz',\
        axis_scale_xy=[1e6,1])

fno[-1]+=1

Nsamps=np.max(np.shape(HDN_ADC1))

# 2021-02-11, apply interleaving error
if flag_ADCInterleav == 'on':

    HDN_intleav = ADC_Interleaving_Error_g(HDN_ADC1, M=M_Interleav, Offset_LevelV=Offset_LevelV,Offset_GaindB=Offset_GaindB,Offset_PhaseDeg=Offset_PhaseDeg)

    PLOT_FFT_dB(HDN_intleav, fs_ADC1, Nsamps=None, plt_format='plot', axis_scale_xy=[1e6,1], color=None, plt_legend=['HD+INTLEAV_'], fnum=fno)

    PLOT_FFT_Info(xRange=[-fs_ADC1/2,fs_ADC1/2],xStep=fs_ADC1/M_Interleav,yRange=plt_yRange_RFBPF,yStep=None,\
        xTicksLabel=None,yTicksLabel=plt_yTicksLabel_HD,\
            xLim=None,yLim=None,\
                xLabel='Frequency(MHz)',yLabel=None,pltLegend=None,pltTitle=str(M_Interleav)+'*ADC Interleaving Offset',\
            axis_scale_xy=[1e6,1])

    fno[-1]+=1

    HDN_ADCOutput = HDN_intleav
else:
    HDN_ADCOutput = HDN_ADC1

# apply NCO
HDN_ADC1_NCO, fNCO = FP_RFSampling_NCO_g(HDN_ADCOutput, fs_ADC1, np.mean(fbwInband_Fund));

HDN_ADC1_NCO_FFT = PLOT_FFT_dB(HDN_ADC1_NCO, fs_ADC1, Nsamps=None, plt_format='plot', axis_scale_xy=[1e6,1], color=None, plt_legend=['HDN_ADC1_NCO_'], fnum=fno)

PLOT_FFT_Info(xRange=[-fs_ADC1/2,fs_ADC1/2],xStep=fs_ADC1/2,yRange=plt_yRange_RFBPF,yStep=None,\
    xTicksLabel=None,yTicksLabel=plt_yTicksLabel_HD,\
        xLim=None,yLim=plt_ylim_RFBPF,\
            xLabel='Frequency(MHz)',yLabel=None,pltLegend=None,pltTitle='NCO: '+str(fNCO/1e6)+'MHz',\
        axis_scale_xy=[1e6,1])

fno[-1]+=1

# generate BBLPF 
if flag_BBLPF == 'on':
    bBB,Ntaps_bBB = FIR_REMEZ(fbands_BBLPF,RippledB_BBLPF,fs_ADC1,'LPF')
    fno[-1]-=1
    bBB_FFT = PLOT_FFT_dB(bBB*Nsamps, fs_ADC1, Nsamps=Nsamps, plt_format='plot_R', axis_scale_xy=[1e6,1], color=None, plt_legend='BB LPF', fnum=fno)
    fno[-1]+=1

    # apply filter by freq. domain
    HDN_bBB = np.fft.ifft((bBB_FFT[0]/Nsamps*HDN_ADC1_NCO_FFT[0]), axis=1)
    # apply filter by time. domain
    # HDN_bRF = np.zeros(np.shape(HDN))
    # for k in range(np.shape(HDN)[0]):
    #     HDN_bRF[k] = np.convolve(HDN[k], bRF, mode='same')
    PLOT_FFT_dB(HDN_bBB, fs_ADC1, Nsamps=None, plt_format='plot', axis_scale_xy=[1e6,1], color=None, plt_legend=['HD+RFBPF_'], fnum=fno)
    
    PLOT_FFT_Info(xRange=[-fs_ADC1/2,fs_ADC1/2],xStep=fs_ADC1/2,yRange=plt_yRange_BBLPF,yStep=None,\
        xTicksLabel=None,yTicksLabel=plt_yTicksLabel_HD,\
            xLim=None,yLim=plt_ylim_BBLPF,\
                xLabel='Frequency(MHz)',yLabel=None,pltLegend=None,pltTitle='NHD+RFBPF',\
            axis_scale_xy=[1e6,1])
    
    fno[-1]+=1

else:
    HDN_bBB = HDN_ADC1_NCO

# apply 2nd sampling by ADC2
HDN_ADC2 = FP_RFSampling_UpDownSampling_wiFIR_g(HDN_bBB, fs_ADC1, fs_ADC2)

PLOT_FFT_dB(HDN_ADC2, fs_ADC2, Nsamps=None, plt_format='plot', axis_scale_xy=[1e6,1], color=None, plt_legend=['HDN_ADC2_'], fnum=fno)

PLOT_FFT_Info(xRange=[-fs_ADC2/2,fs_ADC2/2],xStep=fs_ADC2/2,yRange=plt_yRange_BBLPF,yStep=None,\
    xTicksLabel=None,yTicksLabel=plt_yTicksLabel_HD,\
        xLim=None,yLim=plt_ylim_BBLPF,\
            xLabel='Frequency(MHz)',yLabel=None,pltLegend=None,pltTitle='BBSampling: '+str(fs_ADC2/1e6)+'MHz',\
        axis_scale_xy=[1e6,1])

plt.show()




















