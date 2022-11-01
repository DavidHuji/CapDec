# CapDec: Text-Only Training for Image Captioning using Noise-Injected CLIP


## Official implementation for the paper ["CapDec: Text-Only Training for Image Captioning using Noise-Injected CLIP"](TBD add link) (EMNLP 2022).
![alt text](https://github.com/DavidHuji/CapDec/blob/main/fig1.png)

## Description  
IMPORTANT NOTE: The repo is NOT yet ready. It will be ready in a few days hopefully with a few running examples.
If you still want to use it, please feel free to consult with me in the email below. 

As shown in the paper, CapDec achieves SOTA image-captioning in the setting of training without even a single image.
This is the formal repository for CapDec, in which you can easily reproduce the papers results.

## FlickrStyle7k Examples
Example for styled captions of CapDec on FlickrStyle10K dataset. 
![alt text](https://github.com/DavidHuji/CapDec/blob/main/examples.png)


## Inference Notebooks - TBD
TBD

## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook]&#40;https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing&#41; )
Clone, create environment and install dependencies:  
```
git clone https://github.com/DavidHuji/CapDec && cd CapDec
conda env create -f others/environment.yml
conda activate CapDec
```

# Datasets
1. Download the datasets using the following links: [COCO](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits), [Flickr30K](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits), [FlickrStyle10k](https://zhegan27.github.io/Papers/FlickrStyle_v0.9.zip).
2. Parse the data to the correct format using our script parse_karpathy.py, just make sure to edit head the json paths inside the script.


#Training
Make sure to edit head the json or pkl paths inside the scripts.
1. Extract CLIP features using the following script:
```
python embeddings_generator.py -h
```

2. Training the model using the following script:
```
python train.py --data clip_embeddings_of_last_stage.pkl --out_dir ./coco_train/
```

**There are a few interesting configurable parameters for training as follows:** 
```
output of train.py -h
```

# Evaluation
TBD upload pycocoeval here

# Pre Trained Models
We upload the trained weights that we used for creating Fig.3 in the paper.
[Here](link to weights in drive TBD) are the trained weight of 9 different noise levels. 
You can download it if you do not want to wait for training.

# Open Text Training - Training on any corpus as Harry Potter Books, Shakespeare Plays, or The New York Times (Bonus - Was NOT presented at the paper - )
Cool application of CapDec is to create captions in style of specific corpus that was not even in the form of captions.
Ideally, any given text can be used to train CapDec's decoder to decode CLIP embeddings. It enables to eliminate the need to have any sort of captions textual data. Moreover, it enables to create captioning model that is in the specific style of the given text.
for that, we can first pretrain with images as regular ClipCap, then we fine tune as in CapDec with text only when the text data is a combination of half COCO captions and half sentences from the open text (HP or News) sentences in length between 4 to 20 words.
Here are there a [few nice examples of results.](https://docs.google.com/presentation/d/19WGSbKZKy-Xd3QG4bIR7-zb5t-d_2xkRistgGz0Ykfs/edit#slide=id.gfdad7eec26_0_80)

In order to reproduce that, all you need is to create sentences out of the open text, save them in the right format as the json we have for COCO and then repeat the steps mentioned above for training.
For that you can use the attached script at others/hp_to_coco_format.py.
Although you can use any sort of text for that, you can download the data we used, from the following links: [Harry Potter](https://www.kaggle.com/datasets/balabaskar/harry-potter-books-corpora-part-1-7), [Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays), [News](https://www.kaggle.com/datasets/sbhatti/news-articles-corpus)
You can see an example of the correct format for training at others/parssed_sheikspir_alllines_111k.json

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
