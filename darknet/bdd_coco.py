from argparse import ArgumentParser
from pathlib import Path

import mmcv
from tqdm import tqdm

categories = [{"name": "Bicycle", "id": 1}, {"name": "Car", "id": 2}, {"name": "Pedestrian", "id": 3},
              {"name": "Signal", "id": 4}, {"name": "Signs", "id": 5}, {"name": "Truck", "id": 6}]
attr_id_dict = {"bike": 1, "car": 2, "person": 3, "traffic light": 4, "traffic sign": 5, "truck": 6}

def parse_arguments():
    parser = ArgumentParser(description='BDD100K to COCO format')
    parser.add_argument('--data_dir', type=str, default='./data/bdd')
    return parser.parse_args()


def bdd2coco_detection(labeled_images, fn):
    images = list()
    annotations = list()

    counter = 0
    for i in tqdm(labeled_images):
        counter += 1
        image = dict()
        image['file_name'] = i['name']
        image['height'] = 720
        image['width'] = 1280

        image['id'] = counter

        empty_image = True

        for l in i['labels']:
            annotation = dict()
            if l['category'] in attr_id_dict.keys():
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = image['id']
                x1 = l['box2d']['x1']
                y1 = l['box2d']['y1']
                x2 = l['box2d']['x2']
                y2 = l['box2d']['y2']
                annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                annotation['area'] = float((x2 - x1) * (y2 - y1))
                annotation['category_id'] = attr_id_dict[l['category']]
                annotation['ignore'] = 0
                annotation['id'] = l['id']
                annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                annotations.append(annotation)

        if empty_image:
            continue

        images.append(image)

    attr_dict = {"categories": categories,
                 "images": images,
                 "annotations": annotations,
                 "type": "instances"}
    print('saving...')
    mmcv.dump(attr_dict, fn)


if __name__ == '__main__':
    args = parse_arguments()
    label_dir = Path(args.data_dir) / 'labels'
    save_dir = Path(args.data_dir) / 'coco_annotations'
    save_dir.mkdir()

    print('Loading training set...')
    train_labels = mmcv.load(label_dir / 'bdd100k_labels_images_train.json')
    print('Converting training set...')
    out_fn = save_dir / 'train.json'
    bdd2coco_detection(train_labels, out_fn)

    print('Loading validation set...')
    val_labels = mmcv.load(label_dir / 'bdd100k_labels_images_val.json')
    print('Converting validation set...')
    out_fn = save_dir / 'val.json'
    bdd2coco_detection(val_labels, out_fn)
