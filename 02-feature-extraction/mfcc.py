import librosa
import librosa.display
import numpy as np
from scipy.fftpack import dct

# If you want to see the spectrogram picture
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_spectrogram(spec, note, filepath):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    fig.savefig(filepath, format='png')


def PlotSpecFromWav(y, note):
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(D)
    fig.colorbar(mappable=heatmap)
    librosa.display.specshow(D, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram %s' % note)
    plt.show()


# preemphasis config
alpha = 0.97

# Enframe config
frame_len = 400  # 25ms, fs=16kHz
frame_shift = 160  # 10ms, fs=15kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None)


# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """

    num_samples = signal.size
    num_frames = int(np.floor((num_samples - frame_len) / frame_shift) + 1)
    frames = np.zeros((num_frames, frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift:i * frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win

    return frames


def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2) + 1
    spectrum = np.abs(cFFT[:, 0:valid_len])
    return spectrum


def DefineFilterOnMelScale(num_filter, fmax=8000):
    fmax_mel = Freq2Mel(fmax)

    interval = fmax_mel / (num_filter + 1)
    filter_def = np.zeros((num_filter, 2))
    # (num_filter, [freq_start, freq_end])

    for i in range(filter_def.shape[0]):
        filter_def[i][0] = interval * i
        filter_def[i][1] = interval * (i + 2)

    return filter_def


def Mel2Freq(value_mel):
    value_freq = (10 ** (value_mel / 2595) - 1) * 700
    return value_freq


def Freq2Mel(value_freq):
    return 2595 * np.log10(1 + value_freq / 700)


def MelScale2FreqScale(filter_def_mel):
    filter_def_freq = Mel2Freq(filter_def_mel)
    return filter_def_freq


def BuildMelFilters(filter_def, fft_len, fmax = 8000):
    len = int(fft_len / 2 + 1)
    filters = np.zeros((filter_def.shape[0], len))
    for i in range(filters.shape[0]):
        filters[i] = GetOneFilter(
            filter_def[i][0], filter_def[i][1], fft_len, fmax)
    return filters


def GetOneFilter(start, end, fft_len, fmax):
    from scipy import signal
    len = int(fft_len / 2 + 1)
    num = int(((end - start) / fmax) * len)
    start_point = int((start / fmax) * len)
    end_point = start_point + num

    filter = np.zeros((len))
    filter[start_point:end_point] = signal.triang(num)

    return filter


def GetMelFilters(fft_len, num_filter):
    filter_def = DefineFilterOnMelScale(num_filter)
    filter_def = MelScale2FreqScale(filter_def)
    filter_mel = BuildMelFilters(filter_def, fft_len)
    return filter_mel


def PlotMelFilters(M, note, filepath):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    librosa.display.specshow(M, x_axis='linear')
    plt.ylabel('Mel filter')
    plt.title('Mel filter bank %s' % note)
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(filepath, format='png')


def fbank(spectrum, num_filter=num_filter, norm=None):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """
    filters = GetMelFilters(fft_len, num_filter)
    feats = np.log(np.dot(spectrum, filters.T))
    PlotMelFilters(filters, 'MyFilters', 'filters.png')
    return feats


def mfcc(fbank, num_mfcc=num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    feats = dct(fbank, type=2, axis=1, norm='ortho')[:, 1: (num_mfcc + 1)]

    return feats


def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f = open(file_name, 'w')
    (row, col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i, j]) + ' ')
        f.write(']\n')
    f.close()


def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum)
    mfcc_feats = mfcc(fbank_feats)
    plot_spectrogram(fbank_feats.T, 'Filter Bank', 'fbank.png')
    write_file(fbank_feats, './test.fbank')
    plot_spectrogram(mfcc_feats.T, 'MFCC', 'mfcc.png')
    write_file(mfcc_feats, './test.mfcc')


if __name__ == '__main__':
    main()
