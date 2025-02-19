# MCC: Multi-Cluster Contrastive Semi-Supervised Segmentation Framework for Echocardiogram Videos [IEEE Access]
Official code implementation for the MCC paper accepted by the journal IEEE Access.
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10883951

## Dataset
- MCE_dataset
https://github.com/dewenzeng/MCE_dataset

Folder Structures for MCE Dataset
```
MCE_dataset
|-- images
|   |-- A2C
|   |-- A3C
|   |-- A4C
|   |   |-- subject_000
|   |   |   |-- train_000.png
|   |   |   |-- train_001.png
|   |   |   |-- ...
|   |   |-- subject_001
|   |   |-- ...
|-- labels
|   |-- A2C
|   |-- A3C
|   |-- A4C
|   |   |-- subject_000
|   |   |   |-- train_000.png
|   |   |   |-- train_001.png
|   |   |   |-- ...
|   |   |-- subject_001
|   |   |-- ...
```

- EchoNet-Dynamic Dataset
https://echonet.github.io/dynamic/index.html#dataset

Folder Structures for EchoNet-Dynamic Dataset
```
EchoNet-Dynamic
|-- image
|   |-- 0X10A28877E97DF540
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
|   |-- 0X10B7505562B0A702
|   |-- ...
|-- label
|   |-- 0X10A28877E97DF540
|   |   |-- 0.png
|   |   |-- 1.png
|   |   |-- ...
|   |-- 0X10B7505562B0A702
|   |-- ...
```


- EchoNet-Dynamic Dataset Annotated by Our Professionals
https://drive.google.com/drive/folders/1LkwArTEbBxfSKSv31fLwDXjHjvvyEG4-?usp=sharing

Folder Structures for EchoNet-Dynamic Dataset Annotated by Our Professionals
```
EchoNet-Dynamic_test
|-- Videos
|   |-- 0X1A0A263B22CCD966.avi
|   |-- 0X1A2A76BDB5B98BED.avi
|   |-- ...
|-- FileList.csv
|-- VolumeTracings.csv
```

## Run key selection of the proposed MCC framework
```
cd ./MCE/src_key_selection
python3 train.py --save_dir ../../results/ --src_dir ../../MCE_dataset --key "kmeans" --view "A2C" --img_size 256
```

## Train and Test of the proposed MCC framework with MCE_dataset
```
cd ./MCE/src_xxx
python3 train.py --save_dir ../../results/ --model ${MODEL} --src_dir ../../MCE_dataset --key ${KEY} --view ${VIEW} --batch_size 1 --epochs 50 --img_size 256 --gid "0"
python3 test.py --save_dir ../../results/SS_10%/ --model ${MODEL} --src_dir ../../MCE_dataset --key ${KEY} --view ${VIEW} --batch_size 1 --epochs 50 --img_size 256 --gid "0" --load_model_name "0108-202905" 
```

MODEL: 2DUnet, ConvLSTM, VisTR
VIEW: A2C, A3C, A4C
KEY: kmeans, top3

## Train and Test of the proposed MCC framework with EchoNet-Dynamic Dataset
```
cd ./Echonet/src_xxx
python3 train.py --save_dir ../../results/ --model ${MODEL} --src_dir ../../EchoNet-Dynamic --batch_size 1 --epochs 50 --gid "0"
python3 test.py --save_dir ../../results/SS/ --model ${MODEL} --src_dir ../../EchoNet-Dynamic --batch_size 1 --epochs 50 --gid "0" --load_model_name "0208-021526"
python3 test_extra.py --save_dir ../../results/SS/ --model ${MODEL} --src_dir ../../EchoNet-Dynamic_test --batch_size 1 --epochs 50 --gid "0" --load_model_name "0215-115500" 
```

MODEL: 2DUnet, ConvLSTM, VisTR


## Citation
If you use the code or results in your research, please use the following BibTeX entry.
```
@article{chen2025mcc,
  title={MCC: Multi-Cluster Contrastive Semi-Supervised Segmentation Framework for Echocardiogram Videos},
  author={Chen, Yu-Jen and Lin, Shr-Shiun and Shi, Yiyu and Ho, Tsung-Yi and Xu, Xiaowei},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```
