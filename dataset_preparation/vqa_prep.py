import csv
import json
import shutil
import os
from tqdm import tqdm

answer_type = "other"
question_id = 0
question_type = "what is the image caption"

ans2label = {}
label2ans = []

img_id = 0

for i in range(3):
    data = []

    if i == 0:
        SPLIT = 'train'
    elif i == 1:
        SPLIT = 'test'
    else:
        SPLIT = 'dev'
    
    print(SPLIT)
    print('_______________________________________________')

    with open(f'../../../data/icommongen_v1/genimg_origcap_5i_1c/metadata-{SPLIT}.csv', newline='') as f:
        lines = len(f.readlines())


    with open(f'../../../data/icommongen_v1/genimg_origcap_5i_1c/metadata-{SPLIT}.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',')

        for row in tqdm(reader, total=lines):
            CONCEPT_SET = row[0].replace('$$', ', ')
            sent = 'What is the image caption using concepts : ' + CONCEPT_SET + '?'
            filepath = row[1][0:row[1].rfind('.')]

            if SPLIT == 'test' or SPLIT == 'dev':
                NEW_SPLIT = 'COCO_val2014_'
            else:
                NEW_SPLIT = 'COCO_train2014_'

            label = row[2].replace('""', '')

            dict = {
                "answer_type": answer_type,
                "img_id": NEW_SPLIT + str(img_id),
                "question_id": question_id,
                "question_type": question_type,
                "sent": sent,
                "answers": [{"answer": label, "answer_confidence": "yes", "answer_id": 1}],
                "label": {

                }
            }
            dict["label"][label] = 1
            if not label in ans2label:
                ans2label[label] = question_id
                question_id += 1
            if not label in label2ans:
                label2ans.append(label)
            
            data.append(dict) 

            if i == 1 or i == 2:
                FINAL_SPLIT = 'val2014'
            else:
                FINAL_SPLIT = 'train2014'

            src = '../../../' + row[1]
            dst = f'../../../../py-bottom-up-attention/demo/data/images/{FINAL_SPLIT}'
            shutil.copy(src, dst)

            file_name = row[1][row[1].rfind('/')+1:]
            os.renames(f'{dst}/{file_name}', f'{dst}/{NEW_SPLIT}{img_id}{file_name[1:]}')

            img_id += 1

    if SPLIT == 'dev':
        SPLIT = 'val'
    json.dump(data, open(f'karpathy_{SPLIT}.json', 'w'), indent = 6)

    if SPLIT == 'test':
        NEW_SPLIT = 'val2014'
    elif SPLIT == 'dev':
        continue
    else:
        NEW_SPLIT = 'train2014'
    json.dump({"info": {"description": "This is v2.0 of the VQA dataset.", "url": "http://visualqa.org", "version": "2.0", "year": 2017, "contributor": "VQA Team", "date_created": "2017-04-26 17:00:44"}, "license": {"url": "http://creativecommons.org/licenses/by/4.0/", "name": "Creative Commons Attribution 4.0 International License"}, "data_subtype": NEW_SPLIT,"annotations": data}, open(f'v2_mscoco_{NEW_SPLIT}_annotations.json', 'w'), indent=6)

json.dump(ans2label, open('trainval_ans2label.json', 'w'), indent = 4)
json.dump(label2ans, open('trainval_label2ans.json', 'w'), indent = 4)