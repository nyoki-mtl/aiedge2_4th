# The 2nd AI Edge Contest解法
[第2回AIエッジコンテスト](https://signate.jp/competitions/191)における、チーム「MTL」のFPGA側の提出ソースです。
YOLOv3-tinyのint8量子化と、ソフトウェアにおけるリサイズアルゴリズムの工夫、マルチスレッド処理により
Xilinx DPU IPコア上で40.8FPSという性能を達成しています。

## PL部に関して
ファイル名: design_1.tcl

Vivado 2018.3用のブロックデザイン生成.tclファイルです。
ビットストリーム作成までの手順としては、Ultra96v2向けの空のプロジェクトを作成後、
[公式ページ](https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html#edge)からダウンロード可能なDPU v2.0のIPコアをIP Catalogへ追加します。
その後、.tclファイルを読み込むことでブロックデザインが作成されます。
このブロックデザインに対して、「Create HDL Wrapper」で論理合成・配置配線用のトップモジュールを作成し、
論理合成・配置配線・ビットストリーム生成をおこなっています。

## 学習済ネットワークについて
ファイル名: dpu_yolo_tiny_contest.elf

今回の実装においてはXilinx社のDPU IPコアを用いており、
学習済モデルは.elf形式のオブジェクトファイルとなっています。
C++のアプリケーションソースコードコンパイル時にこのオブジェクトファイルがリンクされ、
実行可能な.elf形式の単一ファイルを生成しています。

## アプリケーションソースコードについて
ファイル名: main.cc, utils.h 

C++で記述された推論アプリケーションのソースコードを添付します。
Xilinx SDK 2018.3環境を利用して、
GNU C++ compiler (aarch64-linux-gnu-g++)でコンパイルをおこない、
アプリケーションを作成しています。
コンパイラとリンカのコマンドラインオプションはそれぞれ次のようなイメージになっています。
```
`aarch64-linux-gnu-g++ -Wall -O3 -g3 -c -fmessage-length=0 -MT"src/main.o" --sysroot=${PetaLinuxで生成したターゲット用のライブラリパスを指定} -MMD -MP -MF"src/main.d" -MT"src/main.o" -o "src/main.o" "../src/main.cc"
```
```
aarch64-linux-gnu-g++ --sysroot=${PetaLinuxで生成したターゲット用のライブラリパスを指定} -o "yolo_tiny_contest.elf"  ./src/main.o  ${.elf形式学習済みモデルのパスを指定} -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lhineon -ln2cube -ldputils -lpthread
```

## 変換スクリプトについて
ファイル名: json_converter.py, fps_calc.py

推論結果のテキストファイルから提出用フォーマットの.jsonファイルを生成するスクリプトと、
複数の画像ごとの推論速度が記述されたテキストファイルから平均FPSを計算するスクリプトです。
