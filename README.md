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

# Reference
If you are interested in the code, please cite our paper:
```
Xi D, Zhuang F, Liu Y, et al. Modelling of bi-directional spatio-temporal dependence and users’ dynamic preferences for missing poi check-in identification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 5458-5465.
```
or in bibtex style:
```
@inproceedings{xi2019modelling,
  title={Modelling of bi-directional spatio-temporal dependence and users’ dynamic preferences for missing poi check-in identification},
  author={Xi, Dongbo and Zhuang, Fuzhen and Liu, Yanchi and Gu, Jingjing and Xiong, Hui and He, Qing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={5458--5465},
  year={2019}
}
```

Last Update Date: July 28, 2019
