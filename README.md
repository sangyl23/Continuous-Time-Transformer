# Continuous-Time-Transformer-For-Channel-Prediction
The pytorch implementation of our paper “Continuous-Time Transformer Based Channel Prediction with Non-Uniform Pilot Pattern”.

<p align="center">
<img align="middle" src="./pictures/Fig4.png" width="800"  />
</p>

## Checkpoint and test data set download
Please download checkpoint (.pth format) and test data set (.mat format) from Tsinghua Cloud: https://cloud.tsinghua.edu.cn/d/b369c2aab6b445e7a067/. Then, extract them in the corresponding `./checkpoint`, `./3GPP_dataset`, and `./deepmimo_dataset` folders.  

## File Description
 - `./3GPP_dataset` containing test dataset of 3GPP TR 38.901 channels.
 - `./deepmimo_dataset` containing test dataset of DeepMIMO channels.
 - `./checkpoint` containing all trained models.
 - `./eval_code` containing all runnable codes for validating the simulation results in the original paper.

Taking the folder `./eval_code/TabIV_attention_omega0_encoding/` as an example (other folders are similar), a detailed overview of the code structure and functionality is shown as below:
 - `./eval_code/TabIV_attention_omega0_encoding/dataloader_ODE.py`: dataset loading file.
 - `./eval_code/TabIV_attention_omega0_encoding/model_ODE.py`: model definition file.
 - `./eval_code/TabIV_attention_omega0_encoding/utils.py`: utility file.
 - `./eval_code/TabIV_attention_omega0_encoding/models/`: folder containing detailed model structure definitions.
 - `./eval_code/TabIV_attention_omega0_encoding/configs/`: folder containing parameter configuration.
 - `./eval_code/TabIV_attention_omega0_encoding/eval_ODE.py`: main file for validation.
 - `./eval_code/TabIV_attention_omega0_encoding/cmd_code.txt`: terminal commands.
 - `./eval_code/TabIV_attention_omega0_encoding/print_results_TabIV.m`: MATLAB code for displaying results.

## Keyword arguments
In our code, we have provided detailed annotations. Below are the specific meanings of some keywords:
 - `b` batch size.
 - `M` BS antenna number.
 - `his_len` estimation sequence length.
 - `pre_len` prediction sequence length.
 - `label_len` label length for transformer decoder.

## Reproduce the experimental result
...
