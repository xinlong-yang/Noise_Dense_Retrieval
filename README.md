# TITAN
Source code for【ICCV2023】Prototypical Mixing and Retrieval-based Refinement for Label Noise-resistant Image Retrieval
![image](https://github.com/xinlong-yang/Noise_Dense_Retrieval/assets/73691354/13dd7f1d-40aa-42a5-bc72-f1a20a97f572)


## Data 
CUB200, CARS196, CIFAR can be downloaded from https://paperswithcode.com/. And we provide CARS98N dataset in the /CARS_98N folder.

## Training
After modify the data path in the 'run.py', use this command in the terminal to train the retrieval model: 'python run.py --train', and the checkpoint, generated hash code will be stored


## Evaluation
After training, modidy the checkpoint file path in the 'run.py', use this command in the terminal to evaluate the trained model: 'python run.py --evaluate'




If you find our work or codebase useful in your research, please cite:
```
@inproceedings{yang2023prototypical,
  title={Prototypical Mixing and Retrieval-based Refinement for Label Noise-resistant Image Retrieval},
  author={Yang, Xinlong and Wang, Haixin and Sun, Jinan and Zhang, Shikun and Chen, Chong and Hua, Xian-Sheng and Luo, Xiao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11239--11249},
  year={2023}
}
```
