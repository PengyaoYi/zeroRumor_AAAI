from collections import defaultdict
import csv
import os
import numpy as np
import random
from random import randint
from transformers import AutoTokenizer
random.seed(42)

def save_dict2npy(np_dict, file_path,mode, train=False):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    np_table = []
    np_false = []
    for key, value in np_dict.items():
        if not train:
            np_table.append(value)
        else:
            if value[1] == '1':
                np_table.append(value)
            else:
                np_false.append(value)
    if train:
        assert len(np_false) - len(np_table) >= 0
        tem_list = []
        for i in range(len(np_false) - len(np_table)):
            a = np_table[randint(0, len(np_table))]
            tem_list.append(a)
        np_table.extend(tem_list)
        assert len(np_false) == len(np_table)
        np_arrray_false = np.array(np_false, dtype=object)
        np.save(file_path + mode + "_false.npy", np_arrray_false)
    np_arrray = np.array(np_table, dtype=object)
    np.save(file_path + mode + ".npy", np_arrray)


tokenizers = AutoTokenizer.from_pretrained("xlm-roberta-base")
class treeNode():
    def __init__(self, line):
        id, parent_id, idx, timestamp, text = line
        self.idx = idx
        self.gid = id
        self.label = None
        self.value = [text, timestamp]
        self.parent = None
        self.parent_id = parent_id if parent_id != 'None' else None
        self.children = []
        self.relation_dict = defaultdict(int)
        self.relation_dict[idx] = 5
        self.abs_relation = None

dataset = "both"
train = False
mode = "dev"
with open("both_data_all.txt", encoding='utf-8') as f:
    data = f.readlines()
with open("both_label_all.txt", encoding='utf-8') as f:
    label = f.readlines()

data_1 = [line.strip().split('\t') for line in data]
lebel_dict = {}
label_1 = [line.strip().split('\t') for line in label]
for line in label_1:
    try:
        lebel_dict[line[0]] = line[1]
    except:
        print(line)
for i, line in enumerate(data_1):
    if line[0] in lebel_dict.keys():
        data_1[i].append(lebel_dict[line[0]])

# with open("ex_data.txt", encoding="utf-8") as f:
#     ex_data = f.readlines()
# ex_data_1 = [line.strip().split('\t') for line in ex_data]
# data_1 = data_1 + ex_data_1

data_2 = []
for i, line in enumerate(data_1):
    if line[0] in lebel_dict.keys():
        data_2.append(line + [lebel_dict[line[0]]])

data_3 = defaultdict(list)
for line in data_2:
    if len(line[4]) < 4:
        print(1)
    data_3[line[0]].append(line[4])
# with open(target_dir + "/test.csv", 'w') as f:
#     writer = csv.writer(f)
#     for key, value in data_3.items():
#         writer.writerow([value[0], lebel_dict[key]])
data_no_comment = defaultdict(list)
for key, value in data_3.items():
    data_no_comment[key] = [value[0], lebel_dict[key]]
save_dict2npy(data_no_comment, './' + dataset + '/no_comment_baseline/', mode, train=train)


# timestamp
# data_5 = defaultdict(list)
# for line in data_2:
#     data_5[line[0]].append([line[3], line[4]])
import copy


# tree
data_7 = {}
for line in data_1:
    if len(line) == 6:
        line = line[:5]
    if line[0] not in data_7.keys():
        data_7[line[0]] = {line[2]: treeNode(line)}
    else:
        data_7[line[0]][line[2]] = treeNode(line)
for key_1 in data_7.keys():
    for id in data_7[key_1].keys():
        if data_7[key_1][id].parent_id is not None:
            data_7[key_1][id].parent = data_7[key_1][data_7[key_1][id].parent_id]
            data_7[key_1][id].relation_dict[data_7[key_1][id].parent_id] = 1
            data_7[key_1][data_7[key_1][id].parent_id].relation_dict[id] = 2
            data_7[key_1][data_7[key_1][id].parent_id].children.append(data_7[key_1][id])
            data_7[key_1][id].abs_relation = data_7[key_1][data_7[key_1][id].parent_id].abs_relation + 1
        else:
            data_7[key_1][id].abs_relation = 0


# 层次遍历
def trans_dict2list(dic, lenth):
    tem_list = [0] * lenth
    for idx in dic.keys():
        tem_list[int(idx) - 1] = dic[idx]
    return tem_list

# import copy
# data_77 = copy.deepcopy(data_7)
from collections import OrderedDict
relation = defaultdict(OrderedDict)
abs_relation = defaultdict(OrderedDict)
data_8 = defaultdict()
count=0
post = 0
time = 0
for key_1 in data_7.keys():
    if key_1 in lebel_dict.keys():
        if len(data_7[key_1]) > 512:
            continue
        #queue = [copy.deepcopy(data_7[key_1]['1'])]
        queue = [data_7[key_1]['1']]
        text = []
        tem_relation = []
        tem_idx = []
        while queue:
            #now_node = queue.pop(0)
            now_node = queue[0]
            tem_idx.append(now_node.idx)
            try:
                tem = data_7[key_1][now_node.idx]
            except:
                print(2)
            queue = queue[1:]
            text.append(now_node.value[0])
            assert int(max(now_node.relation_dict.keys())) <= len(data_7[key_1].keys())
            try:
                relation[key_1][now_node.idx] = trans_dict2list(now_node.relation_dict, len(data_7[key_1].keys()))
            except:
                print(1)
                relation[key_1][now_node.idx] = trans_dict2list(now_node.relation_dict, len(data_7[key_1].keys()))
            abs_relation[key_1][now_node.idx] = now_node.abs_relation
            order_list = []
            for i, child_node in enumerate(sorted(now_node.children, key=lambda d: float(d.value[1]))):
                order_list.append(child_node.idx)
            for i, id in enumerate(order_list):
                for rela_id in order_list[:i]:
                    data_7[key_1][id].relation_dict[rela_id] = 3
                for rela_id in order_list[i + 1:]:
                    data_7[key_1][id].relation_dict[rela_id] = 4
            queue.extend(sorted(now_node.children, key=lambda d: float(d.value[1])))
        count += 1
        input_ids = tokenizers.encode(" <mask> ".join(text))[:512]
        # post += input_ids[::-1].count(250001)
        # time += float(data_7[key_1][tem_idx[input_ids.count(250001)]].value[1])

        data_8[key_1] = [" </s> ".join(text), lebel_dict[key_1], [relation[key_1][item] for item in relation[key_1]], [abs_relation[key_1][item_2] for item_2 in abs_relation[key_1]]]

print(str(post/count))
print(str(time/count))
print("tree_width")
save_dict2npy(data_8,  './' + dataset + '/tree_width/' , mode, train=train)
_time = 0
#time_order

count=0
post = 0
time = 0

idx_order = defaultdict(list)
data_order = defaultdict(list)
for key in data_7.keys():
    if len(data_7[key]) > 512:
        continue
    if key not in lebel_dict.keys():
        continue
    idx_order[key] = sorted(data_7[key], key=lambda d: float(data_7[key][d].value[1]))
    text = []
    tem_relation = []
    tem_abs_relation = []
    for idx in idx_order[key]:
        text.append(data_7[key][idx].value[0])
        try:
            tem_relation.append(relation[key][idx])
        except:
            print(1)
        tem_abs_relation.append(abs_relation[key][idx])
    count += 1
    input_ids = tokenizers.encode(" <mask> ".join(text))[:512]
    post += input_ids[::-1].count(250001)
    # if float(data_7[key][idx_order[key][input_ids.count(250001)]].value[1]) > 970316:
    #     print(1)
    # time += float(data_7[key][idx_order[key][input_ids.count(250001)]].value[1])
    data_order[key] = [" </s> ".join(text), lebel_dict[key], tem_relation, tem_abs_relation]
    # _time += float(data_7[key][idx_order[key][30]].value[1])
save_dict2npy(data_order,  './' + dataset + '/time_order/' , mode, train=train)
print(str(post/count))
print(str(time/count))
print("time_order")
count=0
post = 0
time = 0

idx_invert = defaultdict(list)
data_invert = defaultdict(list)
for key in data_7.keys():
    if len(data_7[key]) > 512:
        continue
    if key not in lebel_dict.keys():
        continue
    idx_invert[key] = sorted(data_7[key], key=lambda d: -float(data_7[key][d].value[1]))
    text = [data_7[key][idx_invert[key].pop(-1)].value[0]]
    tem_relation = [relation[key]['1']]
    tem_abs_relation = [abs_relation[key]['1']]
    for idx in idx_invert[key]:
        text.append(data_7[key][idx].value[0])
        tem_relation.append(relation[key][idx])
        tem_abs_relation.append(abs_relation[key][idx])

    count += 1
    input_ids = tokenizers.encode(" <mask> ".join(text))[:512]
    post += input_ids[::-1].count(250001)
    # if float(data_7[key][idx_order[key][input_ids.count(250001)]].value[1]) > 970316:
    #     print(1)
    time += float(data_7[key][idx_order[key][input_ids.count(250001)]].value[1])

    data_invert[key] = [" </s> ".join(text), lebel_dict[key], tem_relation, tem_abs_relation]
save_dict2npy(data_invert,  './' + dataset + '/time_invert/' , mode, train=train)
# np_table = []
# for key, value in data_8.items():
#     np_table.append(value)
# np_arrray = np.array(np_table, dtype=object)
# np.save('test.npy', np_arrray)
print(str(post/count))
print(str(time/count))
print("time_invert")

# 中序遍历
count=0
post = 0
time = 0

relation_2 = defaultdict(OrderedDict)
abs_relation_2 = defaultdict(OrderedDict)
data_9 = defaultdict(list)
for key_1 in data_7.keys():
    if key_1 in lebel_dict.keys():
        if len(data_7[key_1]) > 512:
            continue
        stack = [copy.deepcopy(data_7[key_1]['1'])]
        # data_7[key_1]['1'].value[0] += " </s> "
        text = []
        tem_idx = []
        while stack:
            now_node = stack.pop(-1)
            text.append(now_node.value[0])
            tem_idx.append(now_node.idx)
            relation_2[key_1][now_node.idx] = trans_dict2list(now_node.relation_dict, len(data_7[key_1].keys()))
            abs_relation_2[key_1][now_node.idx] = now_node.abs_relation
            order_list = []
            for i, child_node in enumerate(sorted(now_node.children, key=lambda d: float(d.value[1]))):
                order_list.append(child_node.idx)
            for i, id in enumerate(order_list):
                for rela_id in order_list[:i]:
                    data_7[key_1][id].relation_dict[rela_id] = 3
                for rela_id in order_list[i + 1:]:
                    data_7[key_1][id].relation_dict[rela_id] = 4
            stack.extend(sorted(now_node.children, key=lambda d: -float(d.value[1])))
        count += 1
        input_ids = tokenizers.encode(" <mask> ".join(text))[:512]
        post += input_ids[::-1].count(250001)
        # if float(data_7[key][idx_order[key][input_ids.count(250001)]].value[1]) > 970316:
        #     print(1)
        time += float(data_7[key_1][tem_idx[input_ids.count(250001)]].value[1])

        data_9[key_1] = [" </s> ".join(text), lebel_dict[key_1], [relation_2[key_1][item] for item in relation_2[key_1]], [abs_relation_2[key_1][item_2] for item_2 in abs_relation_2[key_1]]]
print(1)
print(str(post/count))
print(str(time/count))
print("tree_depth")
save_dict2npy(data_9, './' + dataset + '/tree_depth/' , mode, train=train)
# with open("tree_depth/dev_time_mid.csv", 'w') as f:
#     writer = csv.writer(f)
#     for key, value in data_9.items():
#         writer.writerow(value)



