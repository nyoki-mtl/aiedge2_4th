import json
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm

height, width = 1216, 1936
CLASSES = ('Bicycle', 'Car', 'Pedestrian', 'Signal', 'Signs', 'Truck')
cat_dict = {
    0: 'Bicycle',
    1: 'Car',
    2: 'Pedestrian',
    3: 'Signal',
    4: 'Signs',
    5: 'Truck'
}


def parse():
    parser = ArgumentParser()
    parser.add_argument('--result_path', type=str, default='result.txt')
    parser.add_argument('--sub_path', type=str, default='sub.json')
    return parser.parse_args()


def make_submission(results):
    submission_dict = {}
    for i, line in tqdm(enumerate(results)):
        line = line.rstrip()
        if i % 2 == 0:
            img_id = line
        else:
            if line == '':
                submission_dict[img_id] = {}
            else:
                tmp_res_dict = defaultdict(list)
                res = sorted(eval(line), key=lambda x: x[3], reverse=True)
                for r in res:
                    cat = cat_dict[r[2]]
                    x1 = r[0][0]
                    x2 = r[1][0]
                    y1 = r[0][1]
                    y2 = r[1][1]
                    tmp_res_dict[cat].append([x1, y1, x2, y2])

                res_dict = {}
                for k, v in tmp_res_dict.items():
                    if v: res_dict[k] = v[:100]

                submission_dict[img_id] = res_dict

    # 念の為keyでsort
    submission_dict = dict(sorted(submission_dict.items(), key=lambda x: x[0]))
    return submission_dict


def main():
    args = parse()
    with open(args.result_path) as f:
        results = f.readlines()
    submission_dict = make_submission(results)
    with open(args.sub_path, 'w') as f:
        json.dump(submission_dict, f)


if __name__ == '__main__':
    main()
