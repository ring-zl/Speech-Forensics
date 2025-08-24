import os
import json
import numpy as np
from pprint import pprint
import time
import random

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from libs.aug.augment import call_MixAugment

@register_dataset("PartialSpoof")
class PartialSpoofDataset(Dataset):
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

        # split / training mode
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
        self.crop_ratio = crop_ratio

        self.isAllFeatureMapsUsed = isAllFeatureMapsUsed
        self.featureMapIndex = featureMapIndex

        # load database and select the subset
        if self.is_training:
            label_file =self.train_label_file
        else:
            label_file = self.test_label_file
        dict_db, label_dict = self._load_label_db(label_file)

        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

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

    def get_attributes(self):
        return self.db_attributes
    
    def extract_label_dict(self):
        label_dict = {}
        label_dict["gst-tacotron"] = '0'
        return label_dict
    
    def _load_label_db(self,label_file):

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = self.extract_label_dict()
        print(label_dict)
 
        # load label
        dict_db = tuple()
        with open(label_file ,'r', encoding='utf-8') as file:
            lines = file.readlines()
        for i in range(0, len(lines), 1):
            record = lines[i].strip().split(' ')

            # get key
            key= record[0] 

            if record[2]=='true':
                duration = float(record[1])
                segments = np.empty((0, 2))
                labels = np.empty((0, 1))
            else:
                array_parts = record[3:]
                array_str = ' '.join(array_parts)
                intervals_str = array_str.strip('[]').split('], [')
                intervals = [list(map(float, interval.split(', '))) for interval in intervals_str]
                segments = np.array(intervals).reshape(-1, 2)
                segments = np.asarray(segments, dtype=np.float32)
                duration = float(record[1])
                # get label form dict
                labels =  np.zeros(len(segments))
                labels = (np.asarray(labels, dtype=np.int64))

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
            # print(dict_db)
            # pass
        # pprint(dict_db)
        return dict_db, label_dict

    
    def __len__(self):
        return len(self.data_list)
    
    

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.wav_folder, self.file_prefix + video_item['id'] + self.file_ext)

        if self.file_ext == ".wav":
            waveform, sample_rate = torchaudio.load(filename)
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle_sample_rate)

            # if self.is_training:
            #     waveform = call_MixAugment(waveform, sample_rate=sample_rate, prob=0.5) 
            # start=time.time()
            with torch.inference_mode():
                features, _ = self.model.extract_features(waveform)
            features = torch.stack(features, dim=0).squeeze()
            # end = time.time()
            # print(end-start)


        elif self.file_ext == ".npy":
            features = np.load(filename, mmap_mode='r').astype(np.float32).squeeze()
            features = torch.from_numpy(np.ascontiguousarray(features))

        if self.isAllFeatureMapsUsed:
            feats = features.transpose(2,1)
        else:
            feats = features[self.featureMapIndex].transpose(1,0)
            # feats = features.transpose(1,0)
            # print(feats.shape)

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
    
def random_folder_path(folder_list):
    """
    从提供的文件夹路径列表中随机选择一个路径。

    :param folder_list: 包含文件夹路径的列表。
    :return: 列表中的一个随机路径。
    """
    if not folder_list:
        return None
    return random.choice(folder_list)