import csv
import json

img_ids = {}
sent_ids = {}
img_to_sentids = {}
img_to_sentences = {}

def preprocess(r, s):
    imgid = 0
    sentid = 0
    for row in r:
        filepath = row[1][:((len(row[1])-row[1].rfind('/')-1)*-1)]
        filename = row[1][((len(row[1])-row[1].rfind('/')-1)*-1):]
        img_ids[filepath+filename] = imgid            
        tokens = row[2].lower().split(" ")


        raw = row[2]
        sent_ids[raw] = sentid
        if filepath+filename not in img_to_sentids:
            img_to_sentids[filepath+filename] = []
        img_to_sentids[filepath+filename].append(sentid)
        if filepath+filename not in img_to_sentences:
            img_to_sentences[filepath+filename] = []
        img_to_sentences[filepath+filename].append({
            "tokens": tokens,
            "raw": raw,
            "imgid": img_ids[filepath+filename],
            "sentid": sent_ids[raw]
        })
        imgid += 1
        sentid += 1

data = {
    "images": []
}
cocoid = 0

for i in range(2):
    img_ids.clear()
    sent_ids.clear()
    img_to_sentids.clear()
    img_to_sentences.clear()
    
    if i == 0:
        SPLIT = "train"
    else:
        SPLIT = "test"

    with open(f'../../../data/icommongen_v1/genimg_origcap_5i_1c/metadata-{SPLIT}.csv', newline='') as f:
        r = list(csv.reader(f, delimiter=","))
        preprocess(r, SPLIT)

        j = 0

        for row in r:
            if j > 100:
                break
            if SPLIT == "test":
                real_filepath = "val2014"
            else:
                real_filepath = SPLIT + "2014"
            filepath = row[1][:((len(row[1])-row[1].rfind('/')-1)*-1)]
            filename = row[1][((len(row[1])-row[1].rfind('/')-1)*-1):]
            tokens = row[2].lower().split(" ")
            raw = row[2]

            data["images"].append({
                "filepath": real_filepath,
                "sentids": img_to_sentids[filepath+filename],
                "filename": filename,
                "split": SPLIT,
                "sentences": img_to_sentences[filepath+filename],
                "cocoid": cocoid,
            })
            cocoid += 1
            j += 1

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)