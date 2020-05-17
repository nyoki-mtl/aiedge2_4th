# モデル学習パート

## Data Directroy

aiedgeのデータとBDD100Kを以下のように配置

```
.
└── data
     ├── aiedge
     │   ├── FPGA_seminar_materials
     │   ├── TinyYoloV3_sample
     │   ├── Ultra96-V2
     │   ├── dtc_definition
     │   ├── dtc_evaluation_code
     │   ├── dtc_readme.txt
     │   ├── dtc_sample_submit.json
     │   ├── dtc_test_images
     │   ├── dtc_train_annotations
     │   ├── dtc_train_images
     │   └── submission_details.pdf
     └── bdd
         ├── images
         ├── info
         └── labels
```

COCOに変換
```
$ python aiedge_coco.py --data_dir data/aiedge/
$ python bdd_coco.py --data_dir data/bdd/
```

COCOからYOLO形式に変換
```
$ git clone https://github.com/ssaru/convert2Yolo.git
$ cd convert2yolo
```

aiedge変換
```
$ mkdir ../cfg/aiedge/train
$ mkdir ../cfg/aiedge/val
$ mkdir ../cfg/aiedge/trainval

$ python example.py --dataset COCO --img_type ".jpg" \
--img_path /opt/darknet/cfg/aiedge/datasets \
--cls_list_file ../cfg/aiedge/aiedge.names \
--label ../data/aiedge/coco_annotations/train.json \
--convert_output_path ../cfg/aiedge/train/ \
--manipast_path ../cfg/aiedge/train/
$ python example.py --dataset COCO --img_type ".jpg" \
--img_path /opt/darknet/cfg/aiedge/datasets \
--cls_list_file ../cfg/aiedge/aiedge.names \
--label ../data/aiedge/coco_annotations/val.json \
--convert_output_path ../cfg/aiedge/val/ \
--manipast_path ../cfg/aiedge/val/
$ python example.py --dataset COCO --img_type ".jpg" \
--img_path /opt/darknet/cfg/aiedge/datasets \
--cls_list_file ../cfg/aiedge/aiedge.names \
--label ../data/aiedge/coco_annotations/trainval.json \
--convert_output_path ../cfg/aiedge/trainval/ \
--manipast_path ../cfg/aiedge/trainval/

$ mv ../cfg/aiedge/train/manifast.txt ../cfg/aiedge/train.txt
$ mv ../cfg/aiedge/val/manifast.txt ../cfg/aiedge/val.txt
$ mv ../cfg/aiedge/trainval/manifast.txt ../cfg/aiedge/trainval.txt
$ mv ../cfg/aiedge/trainval ../cfg/aiedge/datasets
$ rm -rf ../cfg/aiedge/train ../cfg/aiedge/val/
$ cp ../data/aiedge/dtc_train_images/*.jpg ../cfg/aiedge/datasets/
$ cp ../data/aiedge/dtc_test_images/*.jpg ../cfg/aiedge/datasets/
$ mkdir ../cfg/aiedge/backup
```

bdd変換
```
$ mkdir ../cfg/bdd/train
$ mkdir ../cfg/bdd/val

$ python example.py --dataset COCO --img_type ".jpg" \
--img_path /opt/darknet/cfg/bdd/datasets \
--cls_list_file ../cfg/bdd/bdd.names \
--label ../data/bdd/coco_annotations/train.json \
--convert_output_path ../cfg/bdd/train/ \
--manipast_path ../cfg/bdd/train/
$ python example.py --dataset COCO --img_type ".jpg" \
--img_path /opt/darknet/cfg/bdd/datasets \
--cls_list_file ../cfg/bdd/bdd.names \
--label ../data/bdd/coco_annotations/val.json \
--convert_output_path ../cfg/bdd/val/ \
--manipast_path ../cfg/bdd/val/

$ mv ../cfg/bdd/train/manifast.txt ../cfg/bdd/train.txt
$ mv ../cfg/bdd/val/manifast.txt ../cfg/bdd/val.txt
$ mv ../cfg/bdd/train ../cfg/bdd/datasets
$ mv ../cfg/bdd/val/* ../cfg/bdd/datasets
$ rm -rf ../cfg/bdd/val
$ cp ../data/bdd/images/100k/train/*.jpg ../cfg/abdd/datasets/
$ cp ../data/bdd/images/100k/val/*.jpg ../cfg/bdd/datasets/
$ mkdir ../cfg/bdd/backup
```

そうするとcfg以下は次のようになっているはず
```
.
├── cfg
│   ├── aiedge
│   │   ├── aiedge.names
│   │   ├── datasets
│   │   ├── train.txt
│   │   ├── trainval.txt
│   │   ├── val.txt
│   │   └── yolov3_tiny_bdd_v3_trainval.cfg
│   └── bdd
│       ├── bdd.names
│       ├── datasets
│       ├── train.txt
│       ├── val
│       ├── val.txt
│       └── yolov3_tiny_v1.cfg

```

## Train

docker build

```
$ docker build -t darknet:latest .
$ sh run_docker.sh
```

以下コンテナ環境の作業

COCO pretrainのモデルから重みをコピー
```
$ ./darknet partial cfg/yolov3-tiny.cfg cfg/aiedge/yolov3-tiny.weights cfg/aiedge/yolov3-tiny.conv.15 15
```
BDDでtrain
```
$ ./darknet detector train ./cfg/bdd/bdd.data ./cfg/bdd/yolov3_tiny.cfg ./cfg/bdd/yolov3-tiny.conv.15 -dont_show
```
bddの重みからaiedgeでtrain
```
$ ./darknet detector train ./cfg/aiedge/aiedge.data ./cfg/aiedge/yolov3_tiny_bdd_v3_trainval.cfg \
  ./cfg/bdd/backup/v1/yolov3_tiny_v1_final.weights -dont_show
```

