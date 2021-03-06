# AAAI19_Bi-STDDP
Keras Implementation of Bi-directional Spatio-Temporal Dependence and Users’ Dynamic Preferences Model

Code for the paper:

Dongbo Xi, Fuzhen Zhuang, Yanchi Liu, Jingjing Gu, Hui Xiong, Qing He: Modelling of Bi-directional Spatio-Temporal Dependence and Users' Dynamic Preferences for Missing POI Check-in Identification. AAAI 2019: 5458-5465

Please cite our AAAI'19 paper if you use our codes. Thanks!

Author: Dongbo Xi

# Requirement
python==2.7  
Keras==2.1.0  
tensorflow-gpu==1.2.1  

# Example to run the codes.
```
python Bi-STDDP.py --embedded_dim 64 --hidden_unit 256 --length 5 --batch_size 256 --dropout 0.5 --lr 0.001 --nb_epoch 50 --earlystop 1 --model_name STDDP.model --dataset NYC 
```

The instruction of commands has been clearly stated in the codes (see the parse_args function).

# Dataset
We use the real-world LBSN datasets from Foursquare. [https://sites.google.com/site/yangdingqi/home/foursquaredataset]

Split the data to train/validation/test files to run the codes directly (see Bi-STDDP-preprocess.py).

Last Update Date: July 28, 2019
