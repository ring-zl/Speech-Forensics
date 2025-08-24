import os
import torchaudio
import numpy as np
import time
import torch
# wav_path = "/data/jzl/SpeechForensics/seen/LJ012-0085.wav"
# bundle = torchaudio.pipelines.WAVLM_BASE
# model = bundle.get_model()
# waveform, sample_rate = torchaudio.load(wav_path)
# waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
# features, _ = model.extract_features(waveform)
# print(features[1].shape)
# pass
import os.path
import sys
import time
from multiprocessing import Process

def extract_features_and_save(input_list, output_folder):
    start = time.time()
    model_name = "WAV2VEC2_BASE"
    bundle = getattr(torchaudio.pipelines, model_name)
    model = bundle.get_model().to("cuda")
    # Ensure output folder exists
    os.makedirs(input_list[0].strip().split("//")[0] +output_folder, exist_ok=True)

    for file_name in input_list:
        if file_name.endswith('.wav'):

            npy_path = file_name.strip().split("//")[0] +output_folder +file_name.strip().split("//")[1]. replace('.wav', '.npy')

            # 如果npy_path已存在，跳过当前迭代
            if os.path.exists(npy_path):
                print(f"File {npy_path} already exists, skipping feature extraction.")
                continue

            wav_path = file_name
            waveform, sample_rate = torchaudio.load(wav_path)
            # Resample audio to the expected sampling rate
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
            # Extract acoustic features
            features, _ = model.extract_features(waveform.to("cuda"))
            new_tensor = torch.stack(features, dim=0).squeeze()
            print(new_tensor.shape)
            # Save features as .npy file
            
            np.save(npy_path, new_tensor.cpu().detach().numpy()[0])
            print(npy_path)
    
    end = time.time()
    print("All done! Processed {} files. Total time: {:0.2f} sec".format(len((input_list)), end - start))



_dir_dataset = '/data/jzl/SpeechForensics/seen/'  # directory of Dataset
_nb_proc = 1 # 处理文件进程数

if __name__ == '__main__':

    l_utt = []
    # os.walk() 函数可以遍历指定目录及其子目录下的所有文件和子目录.
    # 对于每个目录，函数返回一个三元组，分别是当前目录路径、子目录列表、和文件列表
    for r, ds, fs in os.walk(_dir_dataset):
        for f in fs:
            if os.path.splitext(f)[1] != '.wav': continue
            l_utt.append('/'.join([r, f.replace('\\', '/')]))

    # 多进程处理,创建子进程
    nb_utt_per_proc = int(len(l_utt) / _nb_proc)
    l_proc = []
    for i in range(_nb_proc):
        if i == _nb_proc - 1:
            l_utt_cur = l_utt[i * nb_utt_per_proc:]
        else:
            l_utt_cur = l_utt[i * nb_utt_per_proc: (i + 1) * nb_utt_per_proc]
        l_proc.append(Process(target=extract_features_and_save, args=(l_utt_cur,'/wav2vec2_base_channel0/',)))
        print('%d' % i)

    for i in range(_nb_proc):
        l_proc[i].start()
        print('start %d' % i)
    for i in range(_nb_proc):
        l_proc[i].join()

    print("finished!!!!!!!!!")
