
import pandas as pd
from collections import defaultdict
import pickle
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

def gen_scores(predict,durations,threshold,ouputs):

    output_content = []
    for line in durations:
        output_line = ''
        video_id, duration = line.strip().split('.wav')[0], float(line.strip().split('.wav')[1])

        # 找出匹配的CSV行
        if video_id in predict:

            max_score_index = predict[video_id]["score"].index(max(predict[video_id]["score"]))

            if predict[video_id]["score"][max_score_index] > threshold:

                t_start = abs(round(min(max(predict[video_id]["start"][max_score_index].item(), 0), duration), 2))  # 限制在0到duration之间
                t_end = abs(round(min(max(predict[video_id]["end"][max_score_index].item(), 0), duration), 2))    # 限制在0到duration之间

                duration = round(duration, 2)

                # 检查区间占总时长的比例

                if (t_end - t_start) / duration >= 0.95:
                    output_line = f"{video_id} 0.00-{duration}-F {0}"
                else:
                    if t_start == 0 and t_end < duration:
                        output_line = f"{video_id} 0.00-{t_end:.2f}-F/{t_end:.2f}-{duration}-T {0}"
                    elif t_start > 0 and t_end == duration:
                        output_line = f"{video_id} 0.00-{t_start:.2f}-T/{t_start:.2f}-{duration}-F {0}"
                    else:
                        output_line = f"{video_id} 0.00-{t_start:.2f}-T/{t_start:.2f}-{t_end:.2f}-F/{t_end:.2f}-{duration}-T {0}"       
            else:
                output_line = f"{video_id} 0.00-{duration}-T {1}"
        else:
            output_line = f"{video_id} 0.00-{duration}-F {0}"
        output_content.append(output_line)

    with open(output_path, 'w') as file:
        for line in output_content:
            file.write(line + "\n")

    print(f"Output written to {output_path}")


if __name__ == '__main__':
    pkl_path = '/data/jzl/transformer_mulit/ckpt/add2023_reproduce_aug_imporve_trans/eval_results.pkl'
    duration_path = '/data/jzl/transformer_mulit/source_data/duration_test.txt'
    output_path = 'scores.txt'
    threshold = 0.2

    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    with open(duration_path, 'r') as file:
        duration_lines = file.readlines()
    
    data = reorganized(data)
    gen_scores(data, duration_lines, threshold,output_path)

    