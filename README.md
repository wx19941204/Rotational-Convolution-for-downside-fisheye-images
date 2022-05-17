# Rotational-Convolution-for-downside-fisheye-images

This is a Pytorch implemention of Rotational Convolution on UNet（https://github.com/milesial/Pytorch-UNet/tree/master/unet）.  
The dataset includes 10k images from THEODORE("Learning from THEODORE: A Synthetic Omnidirectional Top-View Indoor
Dataset for Deep Transfer Learning"). 

Our dataset can be downloaded at: https://drive.google.com/file/d/1p-L8BnWZPqGNSfFxfSW91NLBZijD2Czx/view?usp=sharing

![image](https://user-images.githubusercontent.com/56708520/168720416-b4fa4783-e2d6-4b73-b7d4-5e561f794694.png)

Ruquire: Pytorch > 1.7, Groupy(https://github.com/tscohen/GrouPy)



```
Train:
1.covert label: run "convert_label_color2id.py"
2.train: run "train_unet.py"
```
```
Evaluate MIOU MPA:
1.load model file（.pth） to "eval_on_val_for_metrics.py" and run.
2. run  "compute_iouPR.py"
```
