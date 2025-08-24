import os
import json
import numpy as np
from pprint import pprint

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
import torchaudio.transforms as T


@register_dataset("wav2vec2")
class WAV2VEC2Dataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode

        ssl_model,
        # feat_folder,     # folder for features
        wav_folder,     # folder for features

        # json_file,       # json file for annotations
        train_label_file,      # label file for wav
        test_label_file, 

        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len

        isAllFeatureMapsUsed,
        featureMapIndex,
        hand_craft,
        open_set,
        priori_dict,
        onlychannel,
    ):
        # file path
        assert os.path.exists(wav_folder) and os.path.exists(train_label_file) and os.path.exists(test_label_file)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.wav_folder = wav_folder

        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.train_label_file = train_label_file
        self.test_label_file = test_label_file

        # training mode
        self.is_training = is_training

        # ssl_model set
        self.ssl_model = ssl_model

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        # {'overflow': 0, 'tacotron2-DCA': 1, 'neural_hmm': 2, 'vits_FreeVC': 3, 'vits': 4, 'tacotron-DDC_ph': 4, 'your_tts': 4}
        # {'DDC_ph_FreeVC': 0, 'glow-tts_FreeVC': 1, 'glow-tts': 2, 'tacotron-DDC_ph': 4, 'overflow': 4, 'vits_FreeVC': 5, 'tacotron2-DCA': 6, 'vits': 4, 'neural_hmm': 8, 'your_tts': 4}
        # {'overflow': 0, 'tacotron2-DCA': 1, 'neural_hmm': 2, 'vits_FreeVC': 3,'DDC_ph_FreeVC': 4,'glow-tts_FreeVC': 4,'glow-tts': 4,'tacotron-DDC_ph':4,'vits': 4,'your_tts': 4}
        self.crop_ratio = crop_ratio

        self.isAllFeatureMapsUsed = isAllFeatureMapsUsed
        self.featureMapIndex = featureMapIndex
        self.hand_craft = hand_craft

        self.open_set = open_set
        self.priori_dict = priori_dict
        self.onlychannel = onlychannel

        # load database and select the subset
        if self.is_training:
            label_file =self.train_label_file
        else:
            label_file = self.test_label_file
        
            
        dict_db = self._load_label_db(label_file)


        get_class = set(self.label_dict.values())
        assert len(get_class) == num_classes
        
        
        # assert len(self.label_dict) == num_classes
        self.data_list = dict_db

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'wav2vec2',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

        self.bundle = getattr(torchaudio.pipelines, f"{self.ssl_model}")
        self.model = self.bundle.get_model()
        self.bundle_sample_rate = self.bundle.sample_rate
        self.transform = None
        
    

        if self.hand_craft is not None:
            if self.hand_craft == 'MFCC':
                self.transform = T.MFCC(
                                        sample_rate = 16000,
                                        n_mfcc=256,
                                        melkwargs={
                                            "n_fft": 2048,
                                            "n_mels": 256,
                                            "win_length": int(25/1000 * 16000),
                                            "hop_length": int(20/1000 * 16000),
                                            "mel_scale": "htk",
                                            },
                                        )
            elif self.hand_craft == 'LFCC':
                self.transform = T.LFCC(
                                        sample_rate=16000,
                                        n_lfcc=256,
                                        speckwargs={
                                        "n_fft": 2048,
                                        "win_length": int(25/1000 * 16000),
                                        "hop_length": int(20/1000 * 16000),
                                        },
                                        )
    def delta(sef,mat):
        '''
        有限差分滤波器，计算它相对于列的一阶导数。
        win: 滤波器组的系数
        '''
        assert mat.ndim == 2
        win = np.array([-1.0, 0.0, 1.0]).reshape(3, 1)
        mat = np.concatenate((mat[:, :1], mat, mat[:, -1:]), axis=-1)
        mat = np.expand_dims(mat, 2)
        mat = np.concatenate((mat[:, :-2], mat[:, 1:-1], mat[:, 2:]), axis=2)
        t, v = mat.shape[:2]
        mat = np.dot(mat.reshape(-1, 3), win).reshape(t, v)
        return mat                

    def get_attributes(self):
        return self.db_attributes
    
    def extract_label_dict(self):
        label_dict = {}  
        current_label = 0  

        with open(self.train_label_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for i in range(0, len(lines), 2):
            if eval(lines[i + 1].strip())[-1]=="True":
                continue
            methods = eval(lines[i + 1].strip())[-1]
            for method in methods:
                if method not in label_dict:
                    label_dict[method]=current_label
                    current_label += 1
        
        return label_dict
    
    def _load_label_db(self,label_file):

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = self.extract_label_dict()
            self.label_dict = label_dict
        if self.open_set:
            self.label_dict = self.priori_dict

        # load label
        dict_db = tuple()
        with open(label_file ,'r', encoding='utf-8') as file:
            lines = file.readlines()
        for i in range(0, len(lines), 2):
            record = lines[i].strip().split('|')

            # get key
            key= record[0] 

            label = eval(lines[i + 1].strip())
            # get video duration if available
            duration = label[0]
            if label[1]=='True':
                segments = np.empty((0, 2))
                labels = np.empty((0, 1))
            else:

                # get segments information
                segments = np.array(label[2]).reshape(-1, 2)
                segments = np.asarray(segments, dtype=np.float32)
            
                # get label form dict
                labels = []
                for item in label[3]:
                    labels.append([self.label_dict[item]])
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            else:
                self.default_fps = 50
                fps = self.default_fps
            
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )
        print(self.label_dict)

        return dict_db

    
    def __len__(self):
        return len(self.data_list)
    
    

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.wav_folder, self.file_prefix + video_item['id'] + self.file_ext)
        # if not os.path.exists(filename):
        #     filename = filename.replace("seen","unseen")
        assert os.path.exists(filename), f"文件 {filename} 不存在。"


        if self.hand_craft is not None :
            waveform, sample_rate = torchaudio.load(filename)
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            crafted_features = self.transform(waveform).squeeze()
            delta1 = self.delta(crafted_features.T)
            delta2 = self.delta(delta1)
            features = np.concatenate((crafted_features.T, delta1, delta2), axis=1)
            features = torch.from_numpy(np.ascontiguousarray(features))
            feats = features.transpose(1,0)
            # print("handcraft")

        else:
            if self.file_ext == ".wav":
                waveform, sample_rate = torchaudio.load(filename)
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)
                with torch.inference_mode():
                    features, _ = self.model.extract_features(waveform)
                features = torch.stack(features, dim=0).squeeze()

            elif self.file_ext == ".npy":
                features = np.load(filename, mmap_mode='r').astype(np.float32).squeeze()
                # features = np.load(filename).astype(np.float32).squeeze(axis=1)
                features = torch.from_numpy(np.ascontiguousarray(features))

            if self.isAllFeatureMapsUsed:
                feats = features.transpose(2,1)
            else:
                if self.onlychannel:
                    feats = features.transpose(1,0)
                    # print("only")
                else:
                    feats = features[self.featureMapIndex].transpose(1,0)

        # feats = add_noise(feats)

        # deal with downsampling (= increased feat stride)
        # feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        # feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        # print(feats.shape)

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps']  / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}
        # pprint(data_dict)
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )
        # print(data_dict["video_id"])
        return data_dict
