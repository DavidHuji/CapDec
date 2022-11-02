import json

data = 'sheikspir_alllines.txt'


def extract_sentences():
    with open(data) as f:
        txt = f.readlines()
    print(txt[:10])
    txt = [t[1:-2].replace(',', '') for t in txt]

    for t in txt[:10]:
        print(t)

    coco_format = []
    for i, sent in enumerate(txt):
        correct_format = {"image_id": i, "caption": sent, "id": i}
        coco_format.append(correct_format)

    # for name in out_names:
    with open('parssed_' + data[:-3] +'json', 'w') as f:
        json.dump(coco_format, f)
    print('size: ',len(coco_format))


extract_sentences()
