from collections import defaultdict
import pickle
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_error_rates(TP, FP, TN, FN):
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # 假正率
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0  # 假负率
    return FPR, FNR

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=int).flatten()
    y_pred = np.array(y_pred, dtype=int).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape of y_true and y_pred do not match")
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, FP, TN, FN

def calculate_f1_score(TP, FP, TN, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, F1


def reorganized(pick_dict):
    organized_data = defaultdict(lambda: {"start": [], "end": [], "score": [], "label": []})
    for i in range(len(pick_dict["video-id"])):
        video_id = pick_dict["video-id"][i]
        organized_data[video_id]["start"].append(pick_dict["t-start"][i])
        organized_data[video_id]["end"].append(pick_dict["t-end"][i])
        organized_data[video_id]["score"].append(pick_dict["score"][i])
        organized_data[video_id]["label"].append(pick_dict["label"][i])

    result = dict(organized_data)
    return result


def cout_F1_EER_with_threshold(predict,gt,threshold):
    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
    for name, label in gt.items():

            if name in predict:
                predictions = np.ones(len(label))

                for i in range(0,len(predict[name]["start"])):
                    if predict[name]["score"][i]>threshold:
                        
                        start_frame = min(max(int(predict[name]["start"][i] / 0.16), 0), len(label) - 1)
                        end_frame = min(max(int(predict[name]["end"][i] / 0.16), 0), len(label) - 1)
                        predictions[start_frame:end_frame + 1] = 0
                    else:
                        continue
            else:
                predictions = np.ones(len(label))

            TP, FP, TN, FN = calculate_metrics(label, predictions)
            total_TP += TP
            total_FP += FP
            total_TN += TN
            total_FN += FN

    precision, recall, f1 = calculate_f1_score(total_TP, total_FP, total_TN, total_FN)
    FPR, FNR = calculate_error_rates(total_TP, total_FP, total_TN, total_FN)
    return precision, recall, f1, FPR, FNR


def get_best_F1(pre,label,lower_bound, upper_bound,step):
    best_F1 = 0
    best_threshold = 0
    precisions = []
    recalls = []
    f1s = []
    thresholds = []
    FPRs = []
    FNRs = []
    for threshold in np.arange(lower_bound, upper_bound,step):

        precision, recall, f1, FPR, FNR = cout_F1_EER_with_threshold(pre,label,threshold)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        thresholds.append(threshold)
        FPRs.append(FPR)
        FNRs.append(FNR)
        TPRs = [1 - x for x in FNRs]

        if f1 > best_F1:
            best_F1 = f1
            best_threshold = threshold
    min_FPR, max_FPR = min(FPRs), max(FPRs)
    FPRs = np.array(FPRs)
    FNRs = np.array(FNRs)
    abs_diffs = np.abs(FPRs - FNRs)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((FNRs[min_index], FPRs[min_index]))
    # eer = brentq(lambda x: 1. - x - interp1d(FPRs, TPRs)(x), 0.06, 0.1)
    # FPRs = np.array(FPRs)
    # FNRs = np.array(FNRs)
    # EER = FPRs[np.nanargmin(np.absolute((FNRs - FPRs)))]
    # print(f'Best eer: {EER}')
    print_metrics(precisions, recalls, f1s, thresholds)
    print(f'Best Threshold: {best_threshold}, Best F1 Score: {best_F1}')
    print(f'Best eer: {eer}')



def print_metrics(precision, recall, f1, thresholds):
    block = ''
    for threshold, p, r, f in zip(thresholds, precision, recall, f1):
        block += '\n|Threshold = {:.2f}: '.format(threshold)
        block += 'Precision = {:>4.2f} (%) '.format(p*100)
        block += 'Recall = {:>4.2f} (%) '.format(r*100)
        block += 'F1 = {:>4.2f} (%) '.format(f*100)
    print(block)


if __name__ == '__main__':
    pkl_path = '/data/jzl/transformer_mulit/ckpt/partialspoof_20240117_10/eval_results.pkl'
    sub_dataset = 'eval'
    

    label_path = '/data/jzl/transformer_mulit/data/PartialSpoof/' + sub_dataset + '_seglab_0.16.npy'

    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    data = reorganized(data)
    labels = np.load(label_path, allow_pickle=True).item()

    get_best_F1(data,labels,0.0,0.8,0.1)
