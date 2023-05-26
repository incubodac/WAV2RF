from wav2rf_functions import *
audio_file = 'test2.wav'
file_path = 'test2.fit'
mv_avg_t,mv_avg = load_file(audio_file,40,10,4,2500)
plot_RF_HR(mv_avg_t,mv_avg,get_heart_rate_data(file_path))