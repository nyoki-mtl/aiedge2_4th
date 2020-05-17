# -*- coding: utf-8 -*-

import json
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()
    parser.add_argument('--text_path', type=str, default='multi_fps.txt')
    parser.add_argument('--type', type=str, default='multi')
    return parser.parse_args()


def calc_fps(results):
    result = results.split('\n')  # 改行で区切る(改行文字そのものは戻り値のデータには含まれない) 
    sum = 0
    for i in range(6355):
        sum = sum + float(result[i*2+1].split(" ")[1])
        print(result[i].split(" ")[1])
    print('fps', sum/6355.0)
    print('time', 6355.0/sum)


def calc_fps_multi(results):
    result = results.split('\n')  # 改行で区切る(改行文字そのものは戻り値のデータには含まれない) 
    sum = 0
    for i in range(212):
        sum = sum + float(result[i].split(" ")[1])
        print(result[i].split(" ")[1])
    print('fps', sum/212.0)
    print('time', 212.0/sum)


def main():
    args = parse()
    with open(args.text_path) as f:
        results = f.read()
        f.close()
    if args.type == 'single':
        calc_fps(results)
    else:
        calc_fps_multi(results)


if __name__ == '__main__':
    main()
