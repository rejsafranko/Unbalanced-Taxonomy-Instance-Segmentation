# Install required Python packages.
python -m pip install -r requirements.txt

# Install DETECTRON2, MASK2FORMER, FC-CLIP in the modules directory.
cd modules
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"

cd ..