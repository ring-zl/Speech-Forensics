import pickle
from collections import defaultdict
import numpy as np

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

def calculate_error_rates(TP, FP, TN, FN):
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # 假正率
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0  # 假负率
    return FPR, FNR


def calculate_per_class(predictions, labels,  classes):
 
    # 为每个类别初始化计数器
    class_counts = {cls: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for cls in classes}

    # 遍历每个帧，更新计数
    for predicted_label, actual_label in zip(predictions, labels):
        for cls in classes:
            if predicted_label == cls and actual_label == cls:
                class_counts[cls]['TP'] += 1
            elif predicted_label == cls and actual_label != cls:
                class_counts[cls]['FP'] += 1
            elif predicted_label != cls and actual_label == cls:
                class_counts[cls]['FN'] += 1
            elif predicted_label != cls and actual_label != cls:
                class_counts[cls]['TN'] += 1

    return class_counts

def calculate_f1_score(TP, FP, TN, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, F1







def cout_with_threshold(predict,gt_path,threshold):
    total_counts = {cls: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for cls in [0,1,2,3,4,5,6,7]}

    uttence_TP = 0
    uttence_FP = 0
    uttence_FN = 0
    uttence_TN = 0
    with open(gt_path ,'r', encoding='utf-8') as file:
        lines = file.readlines()
    for i in range(0, len(lines), 2):
        record = lines[i].strip().split('|')


        name = record[0] 
        label = eval(lines[i + 1].strip())
        duration = label[0]
        frame_count = int(duration / 0.01)

        frame_label = np.full(frame_count, 7)
        # frame_label = np.zeros(frame_count)
        if label[1]=='True':
            frame_label = frame_label
        else:
                # get segments information
            segments = np.array(label[2]).reshape(-1, 2)

            # 遍历时间区间，将对应的帧置为0
            for interval,label in zip(segments, label[3]):
                label_dict = {'overflow': 0, 'tacotron2-DCA': 1, 'neural_hmm': 2, 'vits_FreeVC': 3, 'vits': 4, 'tacotron-DDC_ph': 5, 'your_tts': 6}

                label_num = label_dict[label]


                start_frame = min(max(int(interval[0] / 0.01), 0), len(frame_label) - 1)
                end_frame = min(max(int(interval[1] / 0.01), 0), len(frame_label) - 1)
                frame_label[start_frame:end_frame + 1] = label_num

        # print(frame_label)
        if name in predict:
            predictions = np.full(len(frame_label), 7)

            for i in range(0,len(predict[name]["start"])):
                if predict[name]["score"][i] > threshold:
                        
                    start_frame = min(max(int(predict[name]["start"][i] / 0.01), 0), len(frame_label) - 1)
                    end_frame = min(max(int(predict[name]["end"][i] / 0.01), 0), len(frame_label) - 1)
                    predictions[start_frame:end_frame + 1] = predict[name]["label"][i]
                else:
                    continue
        else:
            predictions = np.full(len(frame_label), 7)

        # print(predictions)
        cout_dict= calculate_per_class(frame_label, predictions,[0,1,2,3,4,5,6,7])
            # 判断标签是否全为7，是则标签为1，否则为0
        
        uttence_label = 1 if all(x == 7 for x in frame_label) else 0
        # 判断预测是否全为7，是则预测值为1，否则为0
        uttence_prediction = 1 if all(x == 7 for x in predictions) else 0

        if uttence_label == 1 and uttence_prediction == 1:
            uttence_TP += 1
        elif uttence_label == 0 and uttence_prediction == 0:
            uttence_TN += 1
        elif uttence_label == 0 and uttence_prediction == 1:
            uttence_FP += 1
        elif uttence_label == 1 and uttence_prediction == 0:
            uttence_FN += 1

        # print(cout_dict)
        for cls in [0,1,2,3,4,5,6,7]:
            total_counts[cls]['TP'] += cout_dict[cls]['TP']
            total_counts[cls]['FP'] += cout_dict[cls]['FP']
            total_counts[cls]['FN'] += cout_dict[cls]['FN']
            total_counts[cls]['TN'] += cout_dict[cls]['TN']
        # total_TP += TP
        # total_FP += FP
        # total_TN += TN
        # total_FN += FN
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_classes = 7  # 类别总数
    total_FPR = 0
    total_FNR = 0
    for cls in [0,1,2,3,4,5,6]:
        precision, recall, f1 = calculate_f1_score(total_counts[cls]['TP'], total_counts[cls]['FP'], total_counts[cls]['TN'], total_counts[cls]['FN'])
        FPR, FNR = calculate_error_rates(total_counts[cls]['TP'], total_counts[cls]['FP'], total_counts[cls]['TN'], total_counts[cls]['FN'])
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_FPR += FPR
        total_FNR += FNR
    
    uttence_FPR, uttence_FNR = calculate_error_rates(uttence_TP,uttence_FP,uttence_TN,uttence_FN)

    average_precision = total_precision / num_classes
    average_recall = total_recall / num_classes
    average_f1 = total_f1 / num_classes
    average_FPR = total_FPR  / num_classes
    average_FNR = total_FNR  / num_classes
    # precision, recall, f1 = calculate_f1_score(total_TP, total_FP, total_TN, total_FN)
    # FPR, FNR = calculate_error_rates(total_TP, total_FP, total_TN, total_FN)
    return average_precision, average_recall, average_f1, average_FPR, average_FNR, uttence_FPR, uttence_FNR


def get_best(pre,label_path,lower_bound, upper_bound,step):
    best_F1 = 0
    best_threshold = 0
    precisions = []
    recalls = []
    f1s = []
    thresholds = []
    FPRs = []
    FNRs = []
    uttence_FPRs = []
    uttence_FNRs = []
    
    for threshold in np.arange(lower_bound, upper_bound,step):

        precision, recall, f1, FPR, FNR ,uttence_FPR, uttence_FNR = cout_with_threshold(pre,label_path,threshold)


        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        thresholds.append(threshold)

        FPRs.append(FPR)
        FNRs.append(FNR)
        uttence_FPRs.append(uttence_FPR)
        uttence_FNRs.append(uttence_FNR)
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

    uttence_FPRs = np.array(uttence_FPRs)
    uttence_FNRs = np.array(uttence_FNRs)
    abs_diffs = np.abs(uttence_FPRs - uttence_FNRs)
    min_index = np.argmin(abs_diffs)
    uttence_eer = np.mean((uttence_FNRs[min_index], uttence_FPRs[min_index]))

    # eer = brentq(lambda x: 1. - x - interp1d(FPRs, TPRs)(x), 0.06, 0.1)
    # FPRs = np.array(FPRs)
    # FNRs = np.array(FNRs)
    # EER = FPRs[np.nanargmin(np.absolute((FNRs - FPRs)))]
    # print(f'Best eer: {EER}')
    print_metrics(precisions, recalls, f1s, thresholds)
    print(f'Best Threshold: {best_threshold}, Best F1 Score: {best_F1}')
    print(f'Best eer: {eer}')
    print(f'Best uttence_eer: {uttence_eer}')




def print_metrics(precision, recall, f1, thresholds):
    block = ''
    for threshold, p, r, f in zip(thresholds, precision, recall, f1):
        block += '\n|Threshold = {:.2f}: '.format(threshold)
        block += 'Precision = {:>4.2f} (%) '.format(p*100)
        block += 'Recall = {:>4.2f} (%) '.format(r*100)
        block += 'F1 = {:>4.2f} (%) '.format(f*100)
    print(block)
























if __name__ == '__main__':
    pkl_path = '/data/jzl/transformer_mulit/ckpt/wav2vec2_20240123_10_MFCC_32/eval_results.pkl'
    sub_dataset = 'dev'
    label_path = '/data/jzl/transformer_mulit/data/'+sub_dataset+'.txt'

    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    data = reorganized(data)
    print(data)
    get_best(data,label_path,0.0,0.8,0.2)