pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnxruntime-gpu==1.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mmdet==2.25.1 mmsegmentation==0.25.0  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lyft_dataset_sdk networkx==2.2 numba==0.53.0 numpy nuscenes-devkit plyfile scikit-image tensorboard trimesh==2.35.39 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r ./requirements.txt
pip install gpustat

git clone git@github.com:open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc4
python setup.py install
cd ..
 
cd ./projects
pip install -v -e .

pip install numpy==1.23.4
pip install yapf==0.40.1
pip install einops
