# Enhancing Generative Commonsense Reasoning Using Image Cues

View Research Poster [Here](https://drive.google.com/file/d/13Zldt4yikAtW-FKYZILMACBbrCPv3j_0/view?usp=sharing).

![poster!](https://cdn.discordapp.com/attachments/960754834984800286/1000624192196198492/SHINE22-Yi-M-PosterFinal.pptx.png)

## Dataset Preparation

```bash
git clone https://github.com/soumyasanyal/iCommongen.git

conda create -n icommongen python=3.6.9
conda activate icommongen

cd src/scripts/dataset_scripts/
srun --gres=gpu:6000:1 --nodelist ink-titan --time 720 python vqa_data.py
```

## Feature Extraction

```bash
#MAKE SURE YOURE ON INK-RUBY
git clone https://github.com/airsplay/py-bottom-up-attention.git
cd py-bottom-up-attention

# Install python libraries
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install detectron2
python setup.py build develop

cd demo/
srun --gres=gpu:8000:1 --nodelist ink-ruby --time 1440 python detectron2_mscoco_proposal_maxnms.py
```

## Running VL-T5

```bash
#clone VL-T5 model
git clone https://github.com/j-min/VL-T5.git

# Create python environment (optional)
conda create -n vlt5 python=3.7
source activate vlt5

# Install python dependencies
pip install -r requirements.txt

# Download T5/BART backbone checkpoint
python download_backbones.py

# For MSCOCO captioning evaluation (optional; for captioning only)
python -c "import language_evaluation; language_evaluation.download('coco')"

#use gdown to install pretrained checkpoints and datasets
pip install gdown
gdown https://drive.google.com/drive/folders/1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph --folder

#Feature conversion (TSV --> H5)
cd feature_extraction/
srun --gres=gpu:6000:1 --nodelist ink-titan --time 360 python tsv_to_h5.py

#FINE TUNE VL-T5 VQA
# Finetuning with 4 gpus
cd ../VL-T5/
bash scripts/VQA_VLT5.sh 4
bash scripts/VQA_VLBart.sh 4
```
