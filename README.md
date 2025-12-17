# Continuous-Time-Transformer-For-Channel-Prediction
The pytorch implementation of our paper “Continuous-Time Transformer Based Channel Prediction with Non-Uniform Pilot Pattern”.

## Checkpoint and test data set download
Please download checkpoint (.pth format) and test data set (.mat format) from Tsinghua Cloud: https://cloud.tsinghua.edu.cn/d/b369c2aab6b445e7a067/. Then, extract them in the corresponding `./checkpoint`, `./3GPP_dataset`, and `deepmimo_dataset` folders.  

## File Description
 - `./3GPP_dataset` containing test dataset of 3GPP TR 38.901 channels.
 - `./deepmimo_dataset` containing test dataset of DeepMIMO channels.
 - `./checkpoint` containing all trained models.
 - `./eval_code` containing all runnable codes for vadidating the simulation results in the original paper.

## Keyword arguments
In our code, we have provided detailed comments. Below are the specific meanings of some keywords:
 - `b` batch size.
 - `his_len` historical sequence length.
 - `BS_num` BS number.
 - `beam_num` beam number.


## Keyword arguments within runnable code files
...

## Reproduce the experimental result
...
