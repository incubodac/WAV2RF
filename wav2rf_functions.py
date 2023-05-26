import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy import interpolate
from scipy.signal import hilbert, butter, filtfilt
from pydub import AudioSegment
import scipy
import fitparse



def load_file(audio_file,window_size,overlap,cutoff_freq,smoothing):
    audio_data, sr = librosa.load(audio_file)
    window_samples  = int(sr * window_size)
    overlap_samples = int(sr * overlap)
    num_windows = len(audio_data) // overlap_samples

    window_size_sec = window_size - overlap
    moving_average = np.zeros(num_windows)
    b, a = butter(4, cutoff_freq / (sr / 2), btype='low')
    
    for i in range(num_windows):
        start = i * overlap_samples
        end = start + window_samples

        window = audio_data[start:end]

        envelope = np.abs(hilbert(window))
        envelope = filtfilt(b, a, envelope)
        spline = interpolate.UnivariateSpline(np.arange(len(envelope)), envelope, s=smoothing)
        interpolated_envelope = spline(np.arange(len(envelope)))
        times = librosa.samples_to_time(np.arange(len(interpolated_envelope)), sr=sr)
        N = len(times)
        T = 1.0 / sr
        x = np.linspace(0.0, N*T, N, endpoint=False)
        yf = scipy.fft.fft(interpolated_envelope)
        xf = scipy.fft.fftfreq(N, T)[:N//2]

        freqs = xf
        start_idx = np.argmax(freqs >= 0.05)
        end_idx = np.argmax(freqs >= 0.6)
        max_amplitude_freq = freqs[start_idx:end_idx][np.argmax(np.abs(yf[start_idx:end_idx]))]
        moving_average[i] = max_amplitude_freq*60
    
    moving_average_times = np.linspace(window_size_sec / 2, len(audio_data) / sr - window_size_sec / 2, len(moving_average))

    return moving_average_times, moving_average

def plot_RF_HR(moving_average_times,moving_average,heart_rate_values):
    fig, ax1 = plt.subplots()
    ax1.plot(moving_average_times,moving_average, color='blue', label='RF')
    ax1.set_ylabel('Respiratory Frequency (RPM)', color='blue')
    ax1.tick_params(axis='y', colors='blue')
    ax1.set_ylim(0, 50) # Plot heart rate curve on existing plot
    ax2 = ax1.twinx()  # Create a secondary y-axis on the right side
    ax2.plot([0 for i in range(110)]+heart_rate_values, color='red', label='Heart Rate')
    ax2.set_ylabel('Heart Rate', color='red')
    ax2.tick_params(axis='y', colors='red')

    # Optional: Add a legend for the heart rate curve
    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    ax1.legend(lines, [line.get_label() for line in lines])
    plt.show()

def get_heart_rate_data(file_path):
    heart_rate_data = []
    
    with fitparse.FitFile(file_path) as fit_file:
        for record in fit_file.get_messages('record'):
            for data in record:
                if data.name == 'heart_rate':
                    heart_rate_data.append(data.value)
    
    return heart_rate_data

if __name__ == '__main__':
    audio_file = 'test2.wav'
    file_path = 'test2.fit'
    mv_avg_t,mv_avg = load_file(audio_file,40,10,4,2500)
    plot_RF_HR(mv_avg_t,mv_avg,get_heart_rate_data(file_path))
