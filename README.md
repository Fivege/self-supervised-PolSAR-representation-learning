# self-supervised-PolSAR-representation-learning
This repository is an implementation of "Exploring PolSAR Images Representation via Self-Supervised Learning and Its Application on Few-Shot Classification", in IEEE GEOSCIENCE AND REMOTE SENSING LETTERS 2022.

# Requirements
-Python3 (3.6)

-Tensorflow (2.3)

# Training
Self-supervised pretrain
    python pretrain.py

Linear evaluation
    python linearevaluation.py
    
When you use your own data, you need to modify the data preprocessing code to adapt to the corresponding data set.
# Citation
    @ARTICLE{9854883,  
    author={Zhang, Wu and Pan, Zongxu and Hu, Yuxin},  
    journal={IEEE Geoscience and Remote Sensing Letters},   
    title={Exploring PolSAR Images Representation via Self-Supervised Learning and Its Application on Few-Shot Classification},   
    year={2022},  
    volume={19},  
    number={},  
    pages={1-5},  
    doi={10.1109/LGRS.2022.3198135}}

# License
Academic use only.
