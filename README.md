# OPNet: Deep Occlusion Perception Network with Boundary Awareness for Amodal Instance Segmentation

This is the PyTorch  implementation of OPNet built on the open-source detectron2.

"OPNet: Deep Occlusion Perception Network with Boundary Awareness for Amodal Instance Segmentation"

<table>
    <tr>
        <td><center><img src="OPNet.png" height="260">



## Installation

```
conda create -n opnet python=3.7 -y
source activate opnet
 
conda install pytorch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
 

pip install ninja yacs cython matplotlib tqdm
pip install opencv-python==4.4.0.40
pip install scikit-image
pip install -r requirments.txt
export INSTALL_DIR=$PWD
 
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
 
cd $INSTALL_DIR
git clone https://github.com/ZitengXue/OPNet.git
cd OPNet/
python3 setup.py build develop
```
