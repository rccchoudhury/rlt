# conda env
conda create --name rlt python=3.10 -y
conda activate rlt

# pip packages
cd /home/${USER}/PycharmProjects/rlt
pip install -r requirements.txt

# decord specific
sudo apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
pip install decord

# download kinetics
cd /lustre/fsw/coreai_nvfm_vfm/$USER/
git clone https://github.com/cvdfoundation/kinetics-dataset.git
cd kinetics-dataset
bash ./k400_downloader.sh
bash ./k400_extractor.sh
python arrange_by_classes.py --path /lustre/fsw/coreai_nvfm_vfm/$USER/kinetics-dataset/k400

# standardize to how rlt wants it to be
cd /home/${USER}/PycharmProjects/rlt
python scripts/make_annot_file.py --video_folder /lustre/fsw/coreai_nvfm_vfm/$USER/kinetics-dataset/k400/videos/train --output_file train.txt
python scripts/make_annot_file.py --video_folder /lustre/fsw/coreai_nvfm_vfm/$USER/kinetics-dataset/k400/videos/val --output_file val.txt
python scripts/make_annot_file.py --video_folder /lustre/fsw/coreai_nvfm_vfm/$USER/kinetics-dataset/k400/videos/test --output_file test.txt