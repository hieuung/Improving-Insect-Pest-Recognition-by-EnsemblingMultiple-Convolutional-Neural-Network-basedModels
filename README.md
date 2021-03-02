# Improving-Insect-Pest-Recognition-by-EnsemblingMultiple-Convolutional-Neural-Network-basedModels

_Project discription ..._

## Requirement
- python 3.7.10
- torch 1.7.1
- numpy 1.19.5
- matplotlib 3.2.2
- TensorboardX 2.0
- sklearn 0.22.2
- skimage 0.16.2
- imblearn 0.4.3
- barbar

## Dataset
In this work, we using two published datasets:
- IP102 (proposed in https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.pdf)
- D0 (proposed in https://www.sciencedirect.com/science/article/abs/pii/S0168169916308833). Download link https://www.dlearningapp.com/web/DLFautoinsects.htm
## Usage
To preproduct the result:
- Download all requirement packages.
- Download dataset
- Prepare the training data using command:
<pre><code>py data_prepare.py -data IP102 -root ...\\ip102_v1.1-002</code></pre>
_Note: Downloaded IP102's root folder directory must contained the archived .tar flie and .txt files of listing samples for training, validating,and testing. Created root folder for training will be placed in the same directory with this code_
<pre><code>py data_prepare.py -data D0 -root ...\\d0</code></pre>
_Note: Downloaded D0's root folder must contained 40 archived .zip files. Created root folder for training will be placed in the same directory_
- Training phase:
  - Training and Testing ResNet50
    <pre><code>python Trainmain.py -data IP102(or D0) -optim Adam -sch expdecay -l2 0.00001 -do 0.3 -predt True -mn resnet -lr 0.0001 -ep 100 -bz 64 -dv cuda</code></pre>
  - Training and Testing RAN
    <pre><code>python Trainmain.py -data IP102(or D0) -optim SGD -sch myScheduler -l2 0.0000 -do 0.0 -predt False -mn residual-attention -lr 0.1 -ep 100 -bz 32 -dv cuda</code></pre>
  - Training and Testing FPN
    <pre><code>python Trainmain.py -data IP102(or D0) -optim Adam -sch expdecay -l2 0.00001 -do 0.0 -predt True -mn fpn -lr 0.0001 -ep 100 -bz 32 -dv cuda</code></pre>
  - Training MMAL-Net
    <pre><code>python train.py -data IP102(or D0) -dv cuda</code></pre>
  - Testing MMAL-Net
    <pre><code>python test.py -data IP102(or D0) -dv cuda</code></pre>
- Ensemble all models with soft voting
    <pre><code>python implement_ensemble.py -data IP102(or D0) -dv cuda</code></pre>