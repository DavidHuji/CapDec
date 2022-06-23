import re, json

sent_size_min, sent_size_max = 4, 20
pt = "harryPotterBooks.txt"


def extract_sentences():
    with open(pt) as f:
        txt = f.readlines()
    new_txt = ''
    for l in txt:
        if l[:4] != "Page":
            new_txt = new_txt +  ' ' + l
    txt = re.sub('[^A-Za-z"" .]+', '', new_txt)  # delete special characters
    print("######################################")
    print("######################################")
    print("######################################")
    print(txt[:1000])
    txt = txt.split('.')

    txt = [t for t in txt if (sent_size_max > len(t.split(' ')) > sent_size_min)]
    print("######################################")
    print("######################################")
    print("######################################")
    for t in txt[:20]:
        print(t)

    coco_format = []
    for i, sent in enumerate(txt):
        correct_format = {"image_id": i, "caption": sent, "id": i}
        coco_format.append(correct_format)

    # for name in out_names:
    with open('parssed_' + pt[:-3] +'json', 'w') as f:
        json.dump(coco_format, f)
    print('size: ',len(coco_format))


extract_sentences()
