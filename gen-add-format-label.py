import numpy as np

gt_path = '/data/jzl/transformer_mulit/data/dev.txt'
result = []

with open(gt_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

for i in range(0, len(lines), 2):
    record = lines[i].strip().split('|')
    name = record[0]
    label = eval(lines[i + 1].strip())
    duration = label[0]
    frame_count = int(duration / 0.01)
    frame_label = np.full(frame_count, 7)  # 或者 np.zeros(frame_count)

    if label[1] == 'True':
        record = f"{name} 0.00-{duration}-T 1"
        result.append(record)
    else:
        segments = np.array(label[2]).reshape(-1, 2)
        segment_info = ""
        prev_end = 0.0

        for j, segment in enumerate(segments):
            start, end = segment
            if start > prev_end:
                segment_info += f"{prev_end}-{start}-T/"
            segment_info += f"{start}-{end}-F"
            if j < len(segments) - 1:
                segment_info += "/"
            prev_end = end

        if end < duration:
            segment_info += f"/{end}-{duration}-T"

        record = f"{name} {segment_info} 0" if segments.size > 0 else f"{name} 0.00-{duration}-T 0"
        result.append(record)

# 将结果写入文件
with open('Dev-addformat.txt', 'w', encoding='utf-8') as f:
    for line in result:
        f.write(line + '\n')
