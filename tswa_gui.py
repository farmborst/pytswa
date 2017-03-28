#!/usr/bin/python
# -*- coding: utf-8 -*-
# ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
from __future__ import print_function, division
try:
    from Tkinter import (Tk, mainloop, StringVar, Label, Button, N, E, S, W, LabelFrame, _setit, BOTH)
    from ttk import Frame
    from tkFileDialog import askopenfilename
    from tkMessageBox import showerror
except:
    from tkinter import (Tk, mainloop, StringVar, Label, Button, N, E, S, W, LabelFrame, _setit, BOTH)
    from tkinter.ttk import Frame
    from tkinter.filedialog import askopenfilename
    from tkinter.messagebox import showerror
from epics import caget
from h5py import File as h5pyFile
from time import sleep, strftime
from datetime import datetime
from threading import Thread
from Queue import Queue
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.pyplot import figure, rcdefaults, rcParams, close
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2TkAgg)
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy.fft import fft, fftfreq, ifft

def h5save(filename, verbose, **namesandvariables):
    ''' save dataset to hdf5 format
    input:
        - desired filename as string
        - names = values of variables to be saved
    return:
        - saves data to "timestamp_filename.hdf5 in working directory"
        - complete filename is returned
    usage:
        1.  recommended
                datadict = {'a' : 2,
                            'b' : 'foo',
                            'c' : 1.337,
                            'd' : [1, 2, 'c']}
                h5save(filename, True. **datadict)
        2.  alternative
                a=2, b='foo', c=1.337, d=[1, 2, 'c']
                h5save(filename, True. a=a, b=b, c=c, d=d)
                accepted datatypes:
                    - int   -> numpy.int64
                    - str   -> str
                    - float -> numpy.float64
                    - list  -> numpy.ndarray of:
                                - np.string__   if >0 string
                                - np.float64    if >0 float
                                - np.int64      if only ints

    '''
    timstamp = strftime('%Y%m%d%H%M%S')
    filename = ''.join([timstamp, '_', filename, '.hdf5'])
    hdf5_fid = h5pyFile(filename, 'w')
    if verbose:
        print('\n==========================================================')
        print('Beginning to save to %s ...' % filename)
        print('\n----------------------------------------------------------')
    for key, value in namesandvariables.iteritems():
            if verbose:
                print('Saving values in %s ... ' % key)
            hdf5_fid.create_dataset(key, data=value)
    if verbose:
        print('\n----------------------------------------------------------')
        print('... finished saving to %s !' % filename)
        print('\n==========================================================')
    hdf5_fid.close()
    return filename


def evaltswa(counts, bunchcurrents, clip=[38460, 87440], effort='FFTW_MEASURE', dump=0, fitorder=1):

    beg, end = clip
    if dump > 0:
        counts = counts[:-dump, beg:end]
        bunchcurrents = bunchcurrents[:-dump]
    else:
        counts = counts[:, beg:end]

    N = shape(counts)[1]      # number of points taken per measurement
    n = len(bunchcurrents)    # number of measurements made
    fs = 1.2495*1e6           # sampling frequency in Hz
    dt = 1/fs
    t = arange(N)*dt*1e3  # time in ms

    bbfbcntsnorm = (counts.T/bunchcurrents).T

    with open('calib_InterpolatedUnivariateSpline.pkl', 'rb') as fh:
        calib = pickle.load(fh)
    bbfbpos = [calib(bbfbcntsnorm[i, :]) for i in range(n)]

    init_pyfftw(bbfbpos[0], effort=effort)
    wisdom = pyfftw.export_wisdom()

    # Turn on the cache for optimum pyfftw performance
    pyfftw.interfaces.cache.enable()

    # create frequency vector
    fd = linspace(0, fs/2/1e3, N/2)

    # prepare frequency filter
    fcent, fsigm = 190, 50
    fleft, fright = fcent - fsigm, fcent + fsigm
    pts_lft = sum(fd < fleft)
    pts_rgt = sum(fd > fright)
    pts_roi = len(fd) - pts_lft - pts_rgt
    frequencyfilter = concatenate((zeros(pts_lft), hanning(pts_roi), zeros(pts_rgt+N/2)))

    # predefine lists
    fftx = empty([n, N], dtype=complex128)
    fftx_clipped = empty([n, N], dtype=complex128)
    fftx_filtered = empty([n, N], dtype=complex128)
    analytic_signal = empty([n, N], dtype=complex128)
    amplitude_envelope = empty([n, N-1])
    instantaneous_phase = empty([n, N])
    instantaneous_frequency = empty([n, N-1])

    for i in range(n):
        # initialise pyfftw for both signals
        myfftw, myifftw = init_pyfftw(bbfbpos[i], wis=wisdom)

        # calculate fft of signal
        fftx[i, :] = myfftw(bbfbpos[i])

        # clip negative frequencies
        fftx_clipped[i, :] = fftx[i, :]
        fftx_clipped[i, N/2+1:] = 0

        # restore lost energy of negative frequencies
        fftx_clipped[i, 1:N/2] *= 2

        # apply frequency filter
        fftx_filtered[i, :] = fftx_clipped[i, :]*frequencyfilter

        # calculate inverse fft (analytical signal) of filtered and positive frequency only fft
        analytic_signal[i, :] = myifftw(fftx_filtered[i, :])
        amplitude_envelope[i, :] = npabs(analytic_signal[i, :])[:-1]
        instantaneous_phase[i, :] = unwrap(angle(analytic_signal[i, :]))
        instantaneous_frequency[i, :] = diff(instantaneous_phase[i, :]) / (2*pi) * fs

    ''' Damping time
    * amplitude damping time only half of center of mass damping time
    * chromaticity dependant
    * in horizontal plane dispersion and energy dependant
    * dephasing
    * landau damping
    * head tail damping
        > analytic two particle modell shows directional interaction from head to tail and position interchange
    '''
    beg, end = 23, 6000
    t2 = linspace(0, t[-1], N-1)[beg:end]
    amplit = [amplitude_envelope[i, beg:end] for i in range(n)]
    signal = [instantaneous_frequency[i, :][beg:end] for i in range(n)]
    fdamp = []
    initialamp, tau_coherent = empty(n), empty(n)
    for i in range(n):
        '''
        ln[A*e^(d*t)] = ln(A) + d*t
        from linear fit: y = m*t + c we gain:
                     A = e^c
                     d = m
        '''
        M = vstack([t2, ones(len(t2))]).T
        tau_inverse, const = linalg.lstsq(M, log(amplit[i]))[0]
        tau_coherent[i] = -1/tau_inverse
        initialamp[i] = exp(const)
        fdamp.append(lambda t, Amplitude=initialamp[i], tau_coherent=tau_coherent[i]: Amplitude*exp(-t/tau_coherent))

    ''' Instantaneous frequency
    * square increase over amplitude
    * frequency is overlayed with synchrotron frequency (~7kHz)
    * filter out synchrotron frequency with a bandstop filter -> tricky (bad snr in fft)
    * fit assumed square funtion -> wrong
    f(amp) = a + b*amp**2 (+ c*amp**4 + d*amp**6)
    amp(t) = A*exp(-t/tau) -> tau
    f(t) = a + b*exp(-2*t/tau) (+ c*exp(-4*t/tau) + d*exp(-6*t/tau))
    '''
    dt = t2[1] - t2[0]   # ms
    def filtersyn(f):
        N = len(f)
        window = hanning(N)
        f = f*window
        fourier = fft(f)
        freqs = fftfreq(N, d=dt)
        fourier[abs(freqs) > 5] = 0
        filtered = ifft(fourier)
        filtered[1:-1] /= window[1:-1]
        return abs(filtered)

    instfreq = []
    for i in range(n):
        #instfreq.append(filtersyn(signal[i]/1e3))
        instfreq.append(noisefilter(t2, signal[i]/1e3, avgpts=30, smoothfac=40000))


    ''' Amplitude dependant tune shift
    '''
    tswa = []
    fitfun = lambda x, a, b: a + b*x
    for i in range(n):
        x = fdamp[i](t2)**2
        y = instfreq[i]
        popt, pcov = scop.curve_fit(fitfun, x, y)
        tswa.append(popt)

    return (t, t2, bbfbpos, amplit, fdamp, signal, instfreq, tswa,
            initialamp, tau_coherent)


def tswaplot(fig, t, t2, bbfbcntsnorm, amplit, fdamp, signal, instfreq, tswa,
             initialamp, tau_coherent):
    i = 0
    ax = fig.add_subplot(221)
    ax.plot(t, bbfbcntsnorm[0][:], label='signal')
    ax.set_ylim([-15, 15])
    ax.set_xlim([0, 5])
    ax.legend(fancybox=True, loc=0).get_frame().set_alpha(.5)
    ax.set_xlabel('time / (ms)')
    ax.set_ylabel('offset / (mm)')

    ax = fig.add_subplot(222)
    ax.plot(t2, amplit[i], label='amplitude')
    ax.plot(t2, fdamp[i](t2), '-r', label='y = A$\cdot$exp(-t/tau )$\n   A = {0:.3}\n   tau = {1:.3}'.format(initialamp[i], tau_coherent[i]))
    ax.legend(fancybox=True, loc=0).get_frame().set_alpha(.5)
    ax.set_xlabel('time / (ms)')
    ax.set_ylabel('betatron amplitude / (mm)')
    ax.set_xlim([0, 5])

    ax = fig.add_subplot(223)
    ax.plot(t2, signal[i]/1e3, '-b', label='instantaneous frequency')
    ax.plot(t2, instfreq[i], '-r', label='synchrotron tune and noise filtered')
    ax.legend(fancybox=True, loc=0).get_frame().set_alpha(.5)
    ax.set_xlabel('time / (ms))')
    ax.set_ylabel('frequency / (kHz)')
    ax.set_xlim([0, 5])

    fitfun = lambda x, a, b: a + b*x
    ax = fig.add_subplot(224)
    ax.plot(fdamp[i](t2)**2, instfreq[i], '-b', label='tuneshift with amplitude')
    ax.plot(fdamp[i](t2)**2, fitfun(fdamp[i](t2)**2, *tswa[i]), '-r', label=('y = a + b$\cdot$x\n    a={0:.4}\n    b={1:.3} Hz/mm$^2$').format(tswa[i][0], tswa[i][1]*1e3))
    ax.legend(fancybox=True, loc=0).get_frame().set_alpha(.5)
    ax.set_xlabel(r'square amplitude / $(mm^2)$')
    ax.set_ylabel('frequency / (kHz)')



def tswa():
    data = [
        'BBQR:X:SB:RAW',             # bbfb sb measurement
        #'BBQR:Y:SB:RAW',             # bbfb sb measurement
        #'BBQR:Z:SB:RAW',             # bbfb sb measurement
        'BBQR:X:SRAM:MEAN',          # bbfb measurement
        #'BBQR:Y:SRAM:MEAN',          # bbfb measurement
        #'BBQR:Z:SRAM:MEAN',          # bbfb measurement
        'BPMZ1D5R:rdX',              # bbfb bpm position
        'BPMZ1D5R:rdY',              # bbfb bpm position
        'PKIK1D1R:set',              # kicker strength
        'PKIK3D1R:set',              # kicker strength
        'CUMZR:rdCur',               # total storage ring current
        'TOPUPCC:rdCurCS',           # single bunch current
        'CUMZR:MBcurrent',           # fillpattern
        'MCLKHX251C:freq'            # masterclock of RF frequency
        ]
    data = dict(zip(data, [[] for i in range(len(data))]))
    time = []
    while 1:
        try:
            time.append(str(datetime.now()))
            for entry in data:
                data[entry].append(caget(entry))
        except:
            showerror(title='epics error', message='error reading epics')
        t, t2, bbfbcntsnorm, amplit, fdamp, signal, instfreq, tswa, initialamp, tau_coherent  = evaltswa(sig, cur)
        q.put(tunstr)
        root.event_generate('<<update_tunstrvar>>', when='tail')
        sleep(1)


def runthread(fun, argstuple):
    # data plotting in new thread to keep gui (main thread&loop) responsive
    t_run = Thread(target=fun, args=argstuple)
    # automatically let die with main thread -> no global stop required
    t_run.setDaemon(True)
    # start thread
    t_run.start()


def initfigs(tabs):
    close('all')
    figs = []
    for tab in tabs:
        # destroy all widgets in fram/tab and close all figures
        for widget in tab.winfo_children():
            widget.destroy()
        fig = figure()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        figs.append(fig)
        toolbar = NavigationToolbar2TkAgg(canvas, tab)
        canvas.get_tk_widget().pack()
        toolbar.pack()
    return figs


if __name__ == '__main__':
    root = Tk()
    root.title('Tuneshift with ampltiude measurement')
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=True)
    lf_settings = LabelFrame(frame, text="Settings", padx=5, pady=5)
    lf_settings.grid(row=0, column=0, sticky=W+E+N+S, padx=10, pady=10)
    lf_results = LabelFrame(frame, text="Results", padx=5, pady=5)
    lf_results.grid(row=1, column=0, rowspan=2, sticky=W+E+N+S, padx=10, pady=10)
    lf_plots = LabelFrame(frame, text="Matplotlib", padx=5, pady=5)
    lf_plots.grid(row=0, column=1, rowspan=2, sticky=W+E+N+S, padx=10, pady=10)

    rcParams['text.usetex'] = True
    rcdefaults()
    params = {'axes.labelsize': 10,
              'axes.titlesize': 10,
              'axes.formatter.limits': [-2, 3],
              'axes.grid': True,
              'figure.figsize': [8, 4.5],
              'figure.dpi': 100,
              'figure.autolayout': True,
              'figure.frameon': False,
              'font.size': 10,
              'font.family': 'serif',
              'legend.fontsize': 10,
              'lines.markersize': 4,
              'lines.linewidth': 1,
              'savefig.dpi': 100,
              'savefig.facecolor': 'white',
              'savefig.edgecolor': 'white',
              'savefig.format': 'pdf',
              'savefig.bbox': 'tight',
              'savefig.pad_inches': 0.05,
              'text.usetex': True,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10}
    rcParams.update(params)
    fig = initfigs([lf_plots])

    q = Queue()
    strvar_tswa = StringVar()
    update_strvar_tswa = lambda event: strvar_tswa.set(q.get())
    root.bind('<<update_strvar_tswa>>', update_strvar_tswa)
    label_tswa = Label(lf_settings, fg='blue', textvariable=strvar_tswa).grid(row=0, column=0)
    label_tswa = Label(lf_results, fg='blue', textvariable=strvar_tswa).grid(row=0, column=0)


    #runthread(tswa)

    mainloop()