from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/aiedge')
    return parser.parse_args()


def get_trainval_df(cfg):
    print('make trainval split...')
    train_annotations_dir = Path(cfg.data_dir) / 'dtc_train_annotations'
    train_annotation_paths = sorted(train_annotations_dir.glob('*.json'))

    frame_infos = []
    for train_annotation_path in train_annotation_paths:
        name = train_annotation_path.stem
        anno = mmcv.load(train_annotation_path)
        attr = anno['attributes']
        frame_infos.append({'name': name,
                            'route': attr['route'],
                            'timeofday': attr['timeofday'],
                            'frameIndex': anno['frameIndex']})

    train_df = pd.DataFrame(frame_infos)

    test_images_dir = Path(cfg.data_dir) / 'dtc_test_images'
    test_img_paths = sorted(test_images_dir.glob('*.jpg'))

    frame_infos = []
    for test_img_path in test_img_paths:
        name = test_img_path.stem
        frame_infos.append({'name': name,
                            'route': None,
                            'timeofday': None,
                            'frameIndex': None})

    test_df = pd.DataFrame(frame_infos)

    df = pd.concat([train_df, test_df], ignore_index=True)

    df_sorted = df.sort_values(by=['route', 'timeofday', 'frameIndex'])

    val_ratio = 0.2
    route_list = sorted(['Saitama', 'Tokyo1', 'Tokyo2'])
    timeofday_list = sorted(['morning', 'day', 'night'])

    is_train_all = []
    for route in route_list:
        for timeofday in timeofday_list:
            situation_size = len(df_sorted[(df_sorted['route'] == route) & (df_sorted['timeofday'] == timeofday)])
            split = int(np.floor(situation_size * (1 - val_ratio)))
            is_train_list = ['train' if i < split else 'val' for i in range(situation_size)]
            is_train_all += is_train_list

    is_test = ['test'] * (len(df_sorted) - len(is_train_all))
    df_sorted = df_sorted.assign(train_val_test=(is_train_all + is_test))

    val_all = []
    for route in route_list:
        for timeofday in timeofday_list:
            situation_size = len(df_sorted[(df_sorted['route'] == route) & (df_sorted['timeofday'] == timeofday)])
            split_size = int(np.floor(situation_size * val_ratio))
            prev_train_size = int(np.floor(situation_size * (1 - val_ratio)))
            k_list = [3 if (i // split_size == 4 and prev_train_size > i) else i // split_size for i in
                      range(situation_size)]
            val_all += k_list

    df_sorted = df_sorted.assign(k=(val_all + is_test))
    df_sorted.loc[df_sorted['k'] == 5, 'k'] = 4
    df = df_sorted.sort_values(by=(['name']))
    df = df.set_index('name')
    print('done!')
    return df


def convert(cfg, df):
    train_annotations_dir = Path(cfg.data_dir) / 'dtc_train_annotations'
    test_images_dir = Path(cfg.data_dir) / 'dtc_test_images'
    dst_dir = Path(cfg.data_dir) / 'coco_annotations'
    dst_dir.mkdir(exist_ok=True)

    cats = ('Bicycle', 'Car', 'Pedestrian', 'Signal', 'Signs', 'Truck')
    cat_ids = {name: i + 1 for i, name in enumerate(cats)}
    cat_info = []
    for i, cat in enumerate(cats):
        cat_info.append({'name': cat, 'id': i + 1})

    anno_paths = sorted(train_annotations_dir.glob('*.json'))
    for split in ['train', 'val', 'trainval']:
        print(f'split: {split}')
        ret = {'images': [], 'annotations': [], "categories": cat_info}
        for anno_path in tqdm(anno_paths):
            image_id = anno_path.stem
            k = df.loc[image_id, 'k']
            if (split == 'train' and k == 4) or (split == 'val' and k in [0, 1, 2, 3]):
                continue

            ret['images'].append({'file_name': f'{image_id}.jpg',
                                  'height': 1216,
                                  'width': 1936,
                                  'id': image_id})

            anno_dict = mmcv.load(anno_path)
            for lbl in anno_dict['labels']:
                category = lbl['category']
                if category not in cats:
                    continue
                bbox = lbl['box2d']
                bbox = [bbox['x1'], bbox['y1'], bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']]
                ret['annotations'].append({'image_id': image_id,
                                           'id': len(ret['annotations']) + 1,
                                           'bbox': bbox,
                                           'category_id': cat_ids[category]})

        mmcv.dump(ret, dst_dir / f'{split}.json')

    # Test
    img_paths = sorted(test_images_dir.glob('*.jpg'))
    print(f'split: test')
    ret = {'images': [], "categories": cat_info}
    for img_path in tqdm(img_paths):
        image_id = img_path.stem
        ret['images'].append({'file_name': img_path.name,
                              'height': 1216,
                              'width': 1936,
                              'id': image_id})

    mmcv.dump(ret, dst_dir / 'test.json')


if __name__ == '__main__':
    cfg = parse()
    df = get_trainval_df(cfg)
    convert(cfg, df)
