# CapDec: Text-Only Training for Image Captioning using Noise-Injected CLIP


## Official implementation for the paper ["CapDec: Text-Only Training for Image Captioning using Noise-Injected CLIP"](TBD add link) (EMNLP 2022).
![alt text](https://github.com/DavidHuji/CapDec/blob/main/fig1.png)

## Description  
IMPORTANT NOTE: The repo is NOT yet ready. It will be ready in a few days hopefully with a few running examples.
If you still want to use it, please feel free to consult with me in the email below. 

As shown in the paper, CapDec achieves SOTA image-captioning in the setting of training without even a single image.
This is the formal repository for CapDec, in which you can easily reproduce the papers results.

## FlickrStyle7k Examples
![alt text](https://github.com/DavidHuji/CapDec/blob/main/examples.png)

<table>
  <tr>
    <td><img src="Images/CONCEPTUAL_01.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_02.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_03.jpg" ></td>
  </tr>
  <tr>
    <td>3D render of a man holding a globe.</td>
     <td>Students enjoing the cherry blossoms</td>
     <td>Green leaf of lettuce on a white plate.</td>
  </tr>
 </table>
 
 <table>
  <tr>
    <td><img src="Images/CONCEPTUAL_04.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_05.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_06.jpg" ></td>
  </tr>
  <tr>
    <td>The hotel and casino on the waterfront. </td>
     <td>The triangle is a symbol of the soul.</td>
     <td>Cartoon boy in the bath.</td>
  </tr>
 </table>


## Inference Notebooks - TBD
TBD

## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook]&#40;https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing&#41; )
Clone, create environment and install dependencies:  
```
git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption
conda env create -f environment.yml
conda activate clip_prefix_caption
```

## Download Data
###COCO: Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/coco/annotations`.

Download [training images](http://images.cocodataset.org/zips/train2014.zip) and [validation images](http://images.cocodataset.org/zips/val2014.zip) and unzip (We use Karpathy et el. split).
### Flickr
TBD
### Flickr7KStyle
TBD


#Training
Extract CLIP features using:
```
python embeddings_generator.py -h
```
Train with fine-tuning of GPT2:
```
python train.py --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/
```

Train only transformer mapping network:
```
python train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40
```

**If you wish to use ResNet-based CLIP:** 

```
python parse_coco.py --clip_model_type RN50x4
```
```
python train.py --only_prefix --data ./data/coco/oscar_split_RN50x4_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40 --is_rn
```

# Evaluation
TBD upload pycocoeval here

# Bonus - Was NOT presented at the paper - Open Text Training - Training on any corpus as Harry Potter Books, Shakespeare Texts, or The New York Times.
Cool application of CapDec is to create captions in style of specific corpus that was not even in the form of captions.
Ideally, any given text can be used to train CapDec's decoder to decode CLIP embeddings. It enables to eliminate the need to have any sort of captions textual data. Moreover, it enables to create captioning model that is in the specific style of the given text.
for that, we can first pretrain with images as regular ClipCap, then we fine tune as in CapDec with text only when the text data is a combination of half COCO captions and half sentences from the open text (HP or News) sentences in length between 4 to 20 words.
Here are there a [few nice examples of results.](https://docs.google.com/presentation/d/19WGSbKZKy-Xd3QG4bIR7-zb5t-d_2xkRistgGz0Ykfs/edit#slide=id.gfdad7eec26_0_80)

In order to reproduce that, all you need is to create sentences out of the open text, save them in the right format as the json we have for COCO and then repeat the steps mentioned above for training.
For that you can use the attached script at others/hp_to_coco_format.py.
Although you can use any sort of text for that, you can download the data we used, from the following links: [Harry Potter](https://www.kaggle.com/datasets/balabaskar/harry-potter-books-corpora-part-1-7), [Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays), [News](https://www.kaggle.com/datasets/sbhatti/news-articles-corpus)

## Citation
If you use this code for your research, please cite:
```
TBD
@article{,
  title={},
  author={},
  journal={},
  year={2021}
}
```

## Acknowledgments
This repository is based on [CLIP](https://github.com/openai/CLIP), [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) and [pycocotools](https://github.com/sks3i/pycocoevalcap) repositories.


## Contact
For any issue please feel free to contact me at: nukraidavid@mail.tau.ac.il.
