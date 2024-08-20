# PATENTCLIP: Multimodal Analyses of Design Patents 
We introduce PATENTCLIP, a multimodal model trained on large-scale design data including all patents from 2007 to 2022.

:black_nib: We incorporate a distribution-aware classification loss and generate detailed multi-view captions for patent images.


<img width="2168" alt="main_fig" src="">

## Data
:green_book: Sample datas can be viewed and download [here]().

## PatentCLIP
:fire: PatentCLIP is based on [CLIP](https://github.com/openai/CLIP), and we use an open source [open_clip](https://github.com/mlfoundations/open_clip) implementation for finetuning and inference.


:hugs: PatentCLIP-ViT-B [checkpoint](https://huggingface.co/patentclip/PatentCLIP_Vit_B)


#### Usage
Load a PatentCLIP model:
```
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:patentclip/PatentCLIP_Vit_B', device=device)
tokenizer = open_clip.get_tokenizer('hf-hub:patentclip/PatentCLIP_Vit_B')
```

#### Multimodal retrieval results 


|                          |          | **Text-Image** |       |       | **Image-Text** |       |       |
|--------------------------|----------|----------------|-------|-------|----------------|-------|-------|
| **Dataset**              | **Backbone** | **R@1**       | **R@5** | **R@10** | **R@1**       | **R@5** | **R@10** |
|    | ResNet50 | 0.09           | 0.25  | 0.34  | 0.08           | 0.24  | 0.33  |
|   **Image-Caption**                            | ResNet101| 0.11           | 0.27  | 0.36  | 0.09           | 0.26  | 0.35  |
|                          | ViT-B-32 | 0.12           | 0.29  | 0.39  | 0.12           | 0.28  | 0.38  |
|                          | ViT-L-14 | **0.18**       | **0.42** | **0.53** | **0.18**       | **0.39** | **0.50** |
