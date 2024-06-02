# setup.sh
sudo apt update
sudo apt install -y libgdal-dev
sudo apt install -y gdal-bin
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install -r requirements.txt
pip install GDAL
pip install git+https://github.com/PatBall1/detectree2.git
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13
pip install detectron2==0.6 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
