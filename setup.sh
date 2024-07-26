cd modules
git clone "https://github.com/facebookresearch/detectron2.git"
git clone "https://github.com/bytedance/fc-clip.git"
cd ..

pip install -r requirements.txt
#pip install -e modules/detectron2

if [ $(ls data/annotations | wc -l) -eq 1 ]; then
    source ./build_datasets.sh
else
    echo "build_datasets.sh not run: There are either zero or more than one files in data/annotations"
fi