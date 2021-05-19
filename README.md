![Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![PyTorch >=1.6](https://img.shields.io/badge/Tensorflow->=1.10-yellow.svg)
# Multi-Level Graph Encoding with Structural-Collaborative Relation Learning for Skeleton-Based Person Re-Identification
By Haocong Rao, Shihao Xu, Xiping Hu, Jun Cheng, Bin Hu. In [IJCAI 2021](.).

## Introduction
This is the official implementation of MG-SCR model presented by "Multi-Level Graph Encoding with Structural-Collaborative Relation Learning for Skeleton-Based Person Re-Identification". The codes are used to reproduce experimental results in the [paper](.).

![image](https://github.com/Kali-Hac/MG-SCR/blob/main/img/overview.png)

Abstract: Skeleton-based person re-identification (Re-ID) is an emerging open topic providing great value for safety-critical applications. 
Existing methods typically extract hand-crafted features or model skeleton dynamics from the trajectory of body joints, while they rarely explore valuable relation information contained in body structure or motion. To fully explore body relations, we construct graphs to model human skeletons from different levels, and for the first time propose a Multi-level Graph encoding approach with Structural-Collaborative Relation learning (MG-SCR) to encode discriminative graph features for person Re-ID.
Specifically, considering that structurally-connected body components are highly correlated in a skeleton, we first propose a *multi-head structural relation layer* to learn different relations of neighbor body-component nodes in graphs, which helps aggregate key correlative features for effective node representations.
Second, inspired by the fact that body-component collaboration in walking usually carries recognizable patterns, we propose a *cross-level collaborative relation layer* to infer collaboration between different level components, so as to capture more discriminative skeleton graph features.
Finally, to enhance graph dynamics encoding, we propose a novel *self-supervised sparse sequential prediction* task for model pre-training, which facilitates encoding high-level graph semantics for person Re-ID. MG-SCR outperforms state-of-the-art skeleton-based methods, and it can achieve superior performance to many multi-modal methods that utilize extra RGB or depth information.

## Requirements
- Python 3.5
- Tensorflow 1.10.0 (GPU)

## Datasets and Models
We provide three already pre-processed datasets (BIWI, IAS, KGBD) with various sequence lengths on <br/>
https://pan.baidu.com/s/1fHMbYZ8H3hZYqrtENupBmA  &nbsp; &nbsp; &nbsp; password：&nbsp;  jtof  <br/>

All the best models reported in our paper can be acquired on <br/> 
https://pan.baidu.com/s/1EhPWg6pJ0Vl4xOh0swQSDQ &nbsp; &nbsp; &nbsp; password：&nbsp; e7oq  <br/> 
Please download the pre-processed datasets ``Datasets/`` and model files ``trained_models/`` into the current directory. <br/>

The original datasets can be downloaded here: [BIWI and IAS-Lab](http://robotics.dei.unipd.it/reid/index.php/downloads), [KGBD](https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor), [KS20.](http://vislab.isr.ist.utl.pt/datasets/#ks20) <br/> 

Note: The access to the Vislab Multi-view KS20 dataset is available upon request. If you have signed the license agreement and been granted the right to use it, please contact me and I will share the pre-processed KS20 data.
 
 
## Usage

To (1) pre-train the MG-SCR model by sparse sequential prediction (SSP) and (2) fine-tune the model for person Re-ID on a specific dataset, simply run the following command: 

```bash
python train.py --dataset BIWI

# Default options: --dataset BIWI --split '' --length 6 --c_lambda 0.3 --task pre --gpu 0
# --dataset [BIWI, IAS, KGBD, KS20]
# --split ['A' (for IAS-A), 'B' (for IAS-B)] 
# --length [4, 6, 8, 10] 
# --task ['pre' (use SSP pre-training), 'none' (no pre-training)]
# --gpu [0, 1, ...]

```
Please see ```train.py``` for more details.

To print evaluation results (Re-ID Confusion Matrix / Rank-n Accuracy / Rank-1 Accuracy / nAUC) of the trained model, run:

```bash
python evaluate.py --dataset BIWI --model_dir trained_models/xx/xx

# Default options: --dataset BIWI --model_dir best --length 6 --gpu 0
# --dataset [BIWI, IAS, KGBD, KS20] 
# --model_dir [best (load the best models), trained_models/xx/xx (directory of model files, e.g., trained_models/best_models/KS20_87.3_95.5)] 
```
 
Please see ```evaluate.py``` for more details.

## Application to Model-Estimated Skeleton Data 
To extend our model to a large RGB-based gait dataset (CASIA B), we exploit pose estimation methods to extract 3D skeletons from RGB videos of CASIA B as follows:
- Step 1: Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Step 2: Extract the 2D human body joints by using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Step 3: Estimate the 3D human body joints by using [3DHumanPose](https://github.com/flyawaychase/3DHumanPose)

We provide already pre-processed skeleton data of CASIA B for Cross-View Evaluation (**CVE**) (f=20/30/40) on &nbsp; &nbsp; &nbsp; https://pan.baidu.com/s/1gDekBzf-3bBVdd0MGL0CvA &nbsp; &nbsp; &nbsp; password：&nbsp;  x3e5 <br/>
Please download the pre-processed datasets into the directory ``Datasets/``. <br/>

## Usage
To (1) pre-train the MG-SCR model by sparse sequential prediction (SSP) and (2) fine-tune the model for person Re-ID on CASIA B under **CVE** setup, simply run the following command:

```bash
python train-CASIA.py --view 0

# Default options: --dataset CASIA_B --split '' --length 20 --c_lambda 0.3 --task pre --gpu 0
# --length [20, 30, 40] 
# --view [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
# --task ['pre' (use SSP pre-training), 'none' (no pre-training)]
# --gpu [0, 1, ...]

```
Please see ```train-CASIA.py``` for more details. <br/>


## Results
![results](img/MG-SCR-results.png)

## License

MG-SCR is released under the MIT License.
