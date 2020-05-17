/*
 * 下記はXilinx社の公開リポジトリ内のソースコードを参考に記述
 * https://github.com/Xilinx/Edge-AI-Platform-Tutorials
 */

/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/


#include <algorithm>
#include <vector>
#include <atomic>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>
#include <numeric>

#include <dnndk/dnndk.h>

#include "utils.h"


using namespace std;
using namespace cv;
using namespace std::chrono;


#define INPUT_NODE "layer0_conv"

int image_num_begin = 0;  // マルチスレッドの暫定推論速度テスト用画像枚数
int image_num_end = 0;  // マルチスレッドの暫定推論速度テスト用画像枚数

int idxInputImage = 0;  // frame index of input video
int idxShowImage = 0;   // next frame index to be displayed
bool bReading = true;   // flag of reding input frame
chrono::system_clock::time_point start_time;  // 簡単のためグローバルに定義

int is_video = 0;  // videoかどうかのフラグ

typedef pair<int, Mat> imagePair;  // マルチスレッド用に定義
typedef pair<int, image> imageYoloPair;  // マルチスレッド用に定義
class paircomp {
    public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) {
            return (n1.first > n2.first);
        }

        return n1.first > n2.first;
    }
};

// mutex for protection of input frames queue
mutex mtxQueueInput;
// mutex for protection of display frmaes queue
mutex mtxQueueShow;
// input frames queue
queue<pair<int, Mat>> queueInput;  // FIFOのキュー
queue<pair<int, image>> queueInputYolo;  // FIFOのキュー
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;  // プライオリティキュー
// display FPS queue
priority_queue<int, vector<int>, greater<int>> queueFPS;  // プライオリティキュー

vector<vector<vector<float>>> results_fps;  // マルチスレッドでの結果検証用
vector<int> results_fps_num;  // マルチスレッドでの結果検証用

/**
 * @brief Feed input frame into DPU for process
 *
 * @param task - pointer to DPU Task for YOLO-v3 network
 * @param frame - pointer to input frame
 * @param mean - mean value for YOLO-v3 network
 *
 * @return none
 */
void setInputImage(DPUTask* task, const Mat& frame, float* mean) {

    Mat img_copy;
    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);
    int size = dpuGetInputTensorSize(task, INPUT_NODE);
    int8_t* data = dpuGetInputTensorAddress(task, INPUT_NODE);

    image img_new = load_image_cv(frame);
    image img_yolo = resize_image_nn(img_new, width, height);  // for Alexey Darknet, high speed operation

    vector<float> bb(size);
    for(int b = 0; b < height; ++b) {
        for(int c = 0; c < width; ++c) {
            for(int a = 0; a < 3; ++a) {
                bb[b*width*3 + c*3 + a] = img_yolo.data[a*height*width + b*width + c];
            }
        }
    }

    float scale = dpuGetInputTensorScale(task, INPUT_NODE);

    for(int i = 0; i < size; ++i) {
        data[i] = int(bb.data()[i]*scale);
        if(data[i] < 0) data[i] = 127;
    }

    free_image(img_new);
    free_image(img_yolo);
}

void setInputImageForYOLO(DPUTask* task, const image& img, float* mean) {

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);
    int size = dpuGetInputTensorSize(task, INPUT_NODE);
    int8_t* data = dpuGetInputTensorAddress(task, INPUT_NODE);

    image img_yolo = resize_image_nn(img, width, height);  // for Alexey Darknetで、高速動作用に作成

    vector<float> bb(size);
    for(int b = 0; b < height; ++b) {
        for(int c = 0; c < width; ++c) {
            for(int a = 0; a < 3; ++a) {
                bb[b*width*3 + c*3 + a] = img_yolo.data[a*height*width + b*width + c];
            }
        }
    }

    float scale = dpuGetInputTensorScale(task, INPUT_NODE);

    for(int i = 0; i < size; ++i) {
        data[i] = int(bb.data()[i]*scale);
        if(data[i] < 0) data[i] = 127;
    }

    free_image(img_yolo);
}

/**
 * @brief Thread entry for reading image frame from the input video file
 *
 * @param fileName - pointer to video file name
 *
 * @return none
 */
void readFrame(const char *fileName) {
    static int loop = 3;  // 3回動画再生したら終了
    VideoCapture video;
    string videoFile = fileName;
    start_time = chrono::system_clock::now();  // 計測スタート

    while (loop>0) {
        //loop--;  // 無限ループ
        if (!video.open(videoFile)) {
            cout<<"Fail to open specified video file:" << videoFile << endl;
            exit(-1);
        }

        while (true) {
            usleep(20000);
            Mat img;
            if (queueInput.size() < 30) {  // queueに30枚以下しか貯まっていなかったら
                if (!video.read(img) ) {  // リード失敗なら終了
                    break;
                }

                mtxQueueInput.lock();
                queueInput.push(make_pair(idxInputImage++, img));  // queueに1フレーム読み込む、idxをインクリメント
                mtxQueueInput.unlock();
            } else {
                usleep(10);  // queueに30枚以上貯まっていたら10usスリープする
            }
        }

        video.release();  // 終了処理
    }

    exit(0);
}

/**
 * @brief 速度計測用画像読み込みスレッドエントリ
 *
 * @param none
 *
 * @return none
 */
void readFrameCheck() {

    // 画像データのパス用変数
    string img_path_prefix = "data/";
    string img_name_prefix = "test_";
    string img_name_suffix = ".jpg";

    // 画像格納用変数
    Mat img[image_num_end - image_num_begin + 1];
    image img_yolo[image_num_end - image_num_begin + 1];

    // 画像のメモリへの読み込み
    for (int i = 0; i < (image_num_end - image_num_begin + 1); i++) {  // 枚数分繰り返し

		// パスの作成
		ostringstream img_num_format;
		int img_num_calc;
		if ((image_num_begin + i) < 6355) {
			img_num_calc = (image_num_begin + i);
		} else {
			img_num_calc = (image_num_begin + i - 6355);
		}
		img_num_format << setfill('0') << std::setw(4) << img_num_calc;
		string img_num = img_num_format.str();
		string img_name = img_name_prefix + img_num + img_name_suffix;
		string img_path = img_path_prefix + img_name;

		// 画像読み込み
		img_yolo[i] = load_image(img[i], img_path);

    }

    // 画像の番号読み込み
    int img_num[image_num_end - image_num_begin + 1];
    for (int i = 0; i < (image_num_end - image_num_begin + 1); i++) {
        if ((image_num_begin + i) < 6355) {
            img_num[i] = (image_num_begin + i);
        } else {
        	img_num[i] = (image_num_begin + i - 6355);
        }
    }

    //cout << "calc start" << endl;
    start_time = chrono::system_clock::now();  // 計測スタート

    for (int i = 0; i < (image_num_end - image_num_begin + 1); i++) {  // 枚数分繰り返し

        while (true) {
            usleep(20000);
            if (queueInput.size() < 10) {  // queueに10枚以下しか貯まっていなかったら
                mtxQueueInput.lock();
                queueInput.push(make_pair(idxInputImage, img[i]));  // queueに1フレーム読み込む
                queueInputYolo.push(make_pair(idxInputImage++, img_yolo[i]));  // queueに1フレーム読み込む、idxをインクリメント
                mtxQueueInput.unlock();
                break;
            } else {
                usleep(10);  // queueに10枚以上貯まっていたら10usスリープする
            }
        }

    }

    // 終了前にテキストファイル書き込み（追記モード）
    ofstream output_file;
    output_file.open("result_multi.txt", ios::app);

    // 各要素の表示
    for (size_t i = 0; i < results_fps.size(); ++i) {

    	ostringstream img_num_format;
    	img_num_format << setfill('0') << std::setw(4) << img_num[results_fps_num[i]];
    	string img_num = img_num_format.str();
    	string img_name = img_name_prefix + img_num + img_name_suffix;

    	output_file << img_name << endl;

    	output_file << "[";
        for (size_t j = 0; j < results_fps.at(i).size(); ++j) {
           	output_file << "([" << (int)results_fps[i][j][0] << ", "
       			        << (int)results_fps[i][j][1] << "], [" << (int)results_fps[i][j][2] << ", " << (int)results_fps[i][j][3] << "], "
       				    << results_fps[i][j][4] << ", " << results_fps[i][j][5] << ")";
           	if (j == (results_fps.at(i).size() - 1)) {
           	} else {
           	    output_file << ", ";
           	}

        }
        output_file << "]" << endl;

    }


    // テキストファイル作成完了
    output_file.close();


    exit(0);
}

/**
 * @brief Thread entry for displaying image frames
 *
 * @param  none
 * @return none
 *
 */
void displayFrame() {
    Mat frame;

    while (true) {
        mtxQueueShow.lock();

        if (queueShow.empty()) {
            mtxQueueShow.unlock();
            usleep(10);
        } else if (idxShowImage == queueShow.top().first) {
            auto show_time = chrono::system_clock::now();
            stringstream buffer;
            frame = queueShow.top().second;

            auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
            buffer << fixed << setprecision(1)
                   << (float)queueShow.top().first / (dura / 1000000.f);
            string a = buffer.str() + " FPS";
            cv::putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 0},1);
            cv::imshow("Quantized YOLOv3-tiny demo, MTLLAB", frame);

            idxShowImage++;
            queueShow.pop();
            mtxQueueShow.unlock();
            if (waitKey(1) == 'q') {  // キーボードのq押されたら終了
                bReading = false;
                exit(0);
            }
        } else {
            mtxQueueShow.unlock();
        }
    }
}

/**
 * @brief FPS測定用スレッドエントリ
 *
 * @param  none
 * @return none
 *
 */
void displayFPS() {

    while (true) {
        mtxQueueShow.lock();

        if (queueFPS.empty()) {
            mtxQueueShow.unlock();
            usleep(10);
        } else if (idxShowImage == queueFPS.top()) {  // 次の画像の番号がqueueに格納されていたら
            auto show_time = chrono::system_clock::now();  // ここまでの時間計測
            if (idxShowImage == ((image_num_end - image_num_begin + 1) - 10)) {  // (総合テスト枚数 - 10)枚目までの段階で、
				stringstream buffer;
				auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
				buffer << fixed << setprecision(2)
					   << (float)queueFPS.top() / (dura / 1000000.f);
				cout << queueFPS.top() << ": " << buffer.str() << " FPS" << endl;  // これまでの画像における平均FPS出力
            }
            // 次の画像へ向けた処理
            idxShowImage++;
            queueFPS.pop();
            mtxQueueShow.unlock();
            if (waitKey(1) == 'q') {  // キーボードのq押されたら終了
                bReading = false;
                exit(0);
            }
        } else {
            mtxQueueShow.unlock();
        }
    }
}

/**
 * @brief Post process after the runing of DPU for YOLO-v3 network
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param frame
 * @param sWidth
 * @param sHeight
 *
 * @return none
 */
void postProcess(DPUTask* task, Mat& frame, int sWidth, int sHeight){

    /*output nodes of YOLO-v3-tiny */
    const vector<string> outputs_node = {"layer15_conv", "layer22_conv"};

    vector<vector<float>> boxes;
    for(size_t i = 0; i < outputs_node.size(); i++){
        string output_node = outputs_node[i];
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);

        /* Store the object detection frames as coordinate information  */
        detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }

    /* Restore the correct coordinate frame of the original image */
    //correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);  // for Origin Darknet

    /* Apply the computation for NMS */
    cout << "boxes size: " << boxes.size() << endl;
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    float h = frame.rows;
    float w = frame.cols;
    for(size_t i = 0; i < res.size(); ++i) {
        float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;
	
	    cout<<res[i][res[i][4] + 6]<<" ";
	    cout<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<endl;

        if(res[i][res[i][4] + 6] > 0.5 ) {
            int type = res[i][4];

            if (type==0) {  // Bycycle、赤
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 0, 255), 1, 1, 0);
            } else if (type==1) {  // Car、青
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 0, 0), 1, 1, 0);
            } else if (type==2) {  // Pedestrian、緑
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0 ,255, 0), 1, 1, 0);
            } else if (type==3) {  // Signal、黄色
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0 ,255, 255), 1, 1, 0);
            } else if (type==4) {  // Signs、紫
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 0, 255), 1, 1, 0);
            } else {  // Truck
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(41 , 66, 115), 1, 1, 0);
            }
        }

    }
}

/**
 * @brief Post process after the runing of DPU for YOLO-v3 network for contest
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param frame
 * @param sWidth
 * @param sHeight
 *
 * @return none
 */
void postProcessContest(DPUTask* task, Mat& frame, int sWidth, int sHeight){

    /*output nodes of YOLO-v3-tiny */
    const vector<string> outputs_node = {"layer15_conv", "layer22_conv"};

    vector<vector<float>> boxes;
    for(size_t i = 0; i < outputs_node.size(); i++){
        string output_node = outputs_node[i];
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);

        /* Store the object detection frames as coordinate information  */
        detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }

    /* Restore the correct coordinate frame of the original image */
    //correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);  // for Origin Darknet

    /* Apply the computation for NMS */
    cout << "boxes size: " << boxes.size() << endl;
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);


    float h = frame.rows;
    float w = frame.cols;
    for(size_t i = 0; i < res.size(); ++i) {
        float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;

        // res[i][4]はクラス番号、これに6を足した要素に対応する確信度が入っている
        cout<<res[i][res[i][4] + 6]<<" ";  // 確信度
        if (res[i][4]==0) {
        	cout<<"Bycycle"<<" ";
        } else if (res[i][4]==1) {
        	cout<<"Car"<<" ";
        } else if (res[i][4]==2) {
        	cout<<"Pedestrian"<<" ";
        } else if (res[i][4]==3) {
        	cout<<"Signal"<<" ";
        } else if (res[i][4]==4) {
        	cout<<"Signs"<<" ";
        } else {
        	cout<<"Truck"<<" ";
        }
        cout<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<endl;

        if(res[i][res[i][4] + 6] > CONF ) {
            int type = res[i][4];

            if (type==0) {  // Bycycle、赤
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 0, 255), 1, 1, 0);
            } else if (type==1) {  // Car、青
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 0, 0), 1, 1, 0);
            } else if (type==2) {  // Pedestrian、緑
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0 ,255, 0), 1, 1, 0);
            } else if (type==3) {  // Signal、黄色
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0 ,255, 255), 1, 1, 0);
            } else if (type==4) {  // Signs、紫
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 0, 255), 1, 1, 0);
            } else {  // Truck
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(41 , 66, 115), 1, 1, 0);
            }

        }

    }
}

/**
 * @brief Post process after the runing of DPU for YOLO-v3 network for contest
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param frame
 * @param sWidth
 * @param sHeight
 * @param results
 *
 * @return none
 */
void postProcessContestCheck(DPUTask* task, Mat& frame, int sWidth, int sHeight, vector<vector<float>>& results){

    /*output nodes of YOLO-v3-tiny */
    const vector<string> outputs_node = {"layer15_conv", "layer22_conv"};

    vector<vector<float>> boxes;
    for(size_t i = 0; i < outputs_node.size(); i++){
        string output_node = outputs_node[i];
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);

        /* Store the object detection frames as coordinate information  */
        detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }

    /* Restore the correct coordinate frame of the original image */
    //correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);  // for Origin Darknet

    /* Apply the computation for NMS */
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    float h = frame.rows;
    float w = frame.cols;
    for (size_t i = 0; i < res.size(); ++i) {

    	vector<float> result;

        float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;

        // 各要素の代入処理
        // res[i][4]はクラス番号、これに6を足した要素に対応する確信度が入っている
        result.push_back((int)xmin);  // 1要素目
        result.push_back((int)ymin);  // 2要素目
        result.push_back((int)xmax);  // 3要素目
        result.push_back((int)ymax);  // 4要素目
        result.push_back(res[i][4]);  // 5要素目（クラス番号）
        result.push_back(res[i][res[i][4] + 6]);  // 6要素目（確信度）

        results.push_back(result);  // 結果を格納

    }
}

/**
 * @brief Post process after the runing of DPU for YOLO-v3 network
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param frame
 * @param sWidth
 * @param sHeight
 *
 * @return none
 */
void postProcessFPS(DPUTask* task, Mat& frame, int sWidth, int sHeight, vector<vector<float>>& results){

    /*output nodes of YOLO-v3-tiny */
    const vector<string> outputs_node = {"layer15_conv", "layer22_conv"};

    vector<vector<float>> boxes;
    for(size_t i = 0; i < outputs_node.size(); i++){
        string output_node = outputs_node[i];
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);

        /* Store the object detection frames as coordinate information  */
        detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }

    /* Restore the correct coordinate frame of the original image */
    //correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);  // for Origin Darknet

    /* Apply the computation for NMS */
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    float h = frame.rows;
    float w = frame.cols;
    for (size_t i = 0; i < res.size(); ++i) {

    	vector<float> result;

        float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;

        // 各要素の代入処理
        // res[i][4]はクラス番号、これに6を足した要素に対応する確信度が入っている
        result.push_back((int)xmin);  // 1要素目
        result.push_back((int)ymin);  // 2要素目
        result.push_back((int)xmax);  // 3要素目
        result.push_back((int)ymax);  // 4要素目
        result.push_back(res[i][4]);  // 5要素目（クラス番号）
        result.push_back(res[i][res[i][4] + 6]);  // 6要素目（確信度）

        results.push_back(result);  // 結果を格納

    }
}

/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param img
 *
 * @return none
 */
void runYOLOContest(DPUTask* task, Mat& img, image& img_yolo) {
    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);

    chrono::system_clock::time_point start, mid_point_0, mid_point, mid_point_1, end;
    start = chrono::system_clock::now();

    mid_point_0 = chrono::system_clock::now();
    auto elapsed_mid_0 = chrono::duration_cast< chrono::microseconds >(mid_point_0 - start).count();
    cout << "system_clock(): " << elapsed_mid_0 <<"us"<< endl;

    /* feed input frame into DPU Task with mean value */
    setInputImageForYOLO(task, img_yolo, mean);

    mid_point = chrono::system_clock::now();
    auto elapsed_mid = chrono::duration_cast< chrono::microseconds >(mid_point - start).count();
    cout << "system_clock()*2 + Preprocessing: " << elapsed_mid <<"us"<< endl;

    /* invoke the running of DPU for YOLO-v3 */
    dpuRunTask(task);

    mid_point_1 = chrono::system_clock::now();
    auto elapsed_mid_1 = chrono::duration_cast< chrono::microseconds >(mid_point_1 - start).count();
    cout << "system_clock()*3 + Preprocessing + Main: " << elapsed_mid_1 <<"us"<< endl;

    postProcessContest(task, img, width, height);

    end = chrono::system_clock::now();
    auto elapsed = chrono::duration_cast< chrono::microseconds >(end - start).count();
    cout << "system_clock()*4 + Preprocessing + Main + Postprocessing: " << elapsed <<"us"<< endl;
}

/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 *
 * @return none
 */
void runYOLOContestCheck(DPUTask* task) {

    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);

    // テキストファイル作成
    ofstream output_file("result.txt");

    // パス用変数
    string img_path_prefix = "data/";
    string img_name_prefix = "test_";
    string img_name_suffix = ".jpg";

    for (int i = 0; i < 6355; i++) {  // 枚数分繰り返し
    	// パスの作成
    	ostringstream img_num_format;
    	img_num_format << setfill('0') << std::setw(4) << i;
    	string img_num = img_num_format.str();
    	string img_name = img_name_prefix + img_num + img_name_suffix;
    	string img_path = img_path_prefix + img_name;

    	// 各画像の検出結果格納用変数
        vector<vector<float>> results;

    	// 画像読み込み
    	Mat img;
        image img_yolo = load_image(img, img_path);

    	//=======================START ここから測定！=======================
    	chrono::system_clock::time_point start, end;
    	start = chrono::system_clock::now();
    	//=======================START ここから測定！=======================

        // リサイズ処理
        setInputImageForYOLO(task, img_yolo, mean);

        // DPUによる推論処理
        dpuRunTask(task);

        // NMSと結果格納
        postProcessContestCheck(task, img, width, height, results);

        //=======================STOP ここで測定終了！=======================
        end = chrono::system_clock::now();
        auto elapsed = chrono::duration_cast< chrono::microseconds >(end - start).count();
        cout << i << ": " << elapsed << " us" << endl;  // 1枚あたりの推論時間出力
        stringstream buffer;
		buffer << fixed << setprecision(2) << (1000000.f / elapsed);
		cout << i << ": " << buffer.str() << " FPS" << endl;  // 1枚あたりのFPS出力
        //=======================STOP ここで測定終了！=======================

        // メモリ解放
        free_image(img_yolo);

        // テキストファイル書き込み
        // 画像ファイル名
        output_file << img_name << endl;

        // 各要素の表示
        output_file << "[";
        for (size_t i = 0; i < results.size(); ++i) {
        	output_file << "([" << (int)results[i][0] << ", "
        			<< (int)results[i][1] << "], [" << (int)results[i][2] << ", " << (int)results[i][3] << "], "
    				<< results[i][4] << ", " << results[i][5] << ")";
        	if (i == (results.size() - 1)) {
        	} else {
        		output_file << ", ";
        	}

        }
        output_file << "]" << endl;

    }

    // テキストファイル作成完了
    output_file.close();
}

/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 *
 * @return none
 */
void runYOLOValidationCheck(DPUTask* task) {

    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);

    // テキストファイル作成
    ofstream output_file("result_val.txt");

    // パス用変数
    string img_path_prefix = "validation/";
    string img_name_prefix = "train_";
    string img_name_suffix = ".jpg";

    // バリデーションする画像番号を格納
    ifstream ifs("validation_image_num.txt");
    int img_num_valid[4254];
    for (int i=0; i < 4254; i++) {
    	string str;
    	getline(ifs, str);
    	img_num_valid[i] =  atoi(str.c_str());
    }


    for (int i = 0; i < 4254; i++) {  // 枚数分繰り返し
    	// パスの作成
    	ostringstream img_num_format;
    	img_num_format << setfill('0') << std::setw(5) << img_num_valid[i];
    	string img_num = img_num_format.str();
    	string img_name = img_name_prefix + img_num + img_name_suffix;
    	string img_path = img_path_prefix + img_name;

    	// 各画像の検出結果格納用変数
        vector<vector<float>> results;

    	// 画像読み込み
    	Mat img;
        image img_yolo = load_image(img, img_path);

    	//=======================START ここから測定！=======================
    	chrono::system_clock::time_point start, end;
    	start = chrono::system_clock::now();
    	//=======================START ここから測定！=======================

    	// リサイズ処理
        setInputImageForYOLO(task, img_yolo, mean);

        // DPUによる推論処理
        dpuRunTask(task);

        // NMSと結果格納
        postProcessContestCheck(task, img, width, height, results);

        //=======================STOP ここで測定終了！=======================
        end = chrono::system_clock::now();
        auto elapsed = chrono::duration_cast< chrono::microseconds >(end - start).count();
        cout << i << ": " << elapsed << " us" << endl;  // 1枚あたりの推論時間出力
        stringstream buffer;
		buffer << fixed << setprecision(2) << (1000000.f / elapsed);
		cout << i << ": " << buffer.str() << " FPS" << endl;  // 1枚あたりのFPS出力
        //=======================STOP ここで測定終了！=======================

        // メモリ解放
        free_image(img_yolo);

        // テキストファイル書き込み
        // 画像ファイル名
        output_file << img_name << endl;

        // 各要素の表示
        output_file << "[";
        for (size_t i = 0; i < results.size(); ++i) {
        	output_file << "([" << (int)results[i][0] << ", "
        			<< (int)results[i][1] << "], [" << (int)results[i][2] << ", " << (int)results[i][3] << "], "
    				<< results[i][4] << ", " << results[i][5] << ")";
        	if (i == (results.size() - 1)) {
        	} else {
        		output_file << ", ";
        	}

        }
        output_file << "]" << endl;

    }

    // テキストファイル作成完了
    output_file.close();
}


/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 *
 * @return none
 */
void runYOLO_video(DPUTask* task) {
    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);

    while (true) {
        pair<int, Mat> pairIndexImage;

        mtxQueueInput.lock();
        if (queueInput.empty()) {
            mtxQueueInput.unlock();
            if (bReading)
            {
                continue;
            } else {
                break;
            }
        } else {
            /* get an input frame from input frames queue */
            pairIndexImage = queueInput.front();
            queueInput.pop();
            mtxQueueInput.unlock();
        }
        //vector<vector<float>> res;
        /* feed input frame into DPU Task with mean value */
        setInputImage(task, pairIndexImage.second, mean);

        /* invoke the running of DPU for YOLO-v3 */
        dpuRunTask(task);

        postProcess(task, pairIndexImage.second, width, height);

        /* push the image into display frame queue */
        mtxQueueShow.lock();
        queueShow.push(pairIndexImage);  // 結果をqueueへ格納
        mtxQueueShow.unlock();
    }
}

/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - FPS測定
 *
 * @return none
 */
void runYOLO_FPS(DPUTask* task) {
    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);

    while (true) {
        pair<int, Mat> pairIndexImage;
        pair<int, image> pairIndexImageYolo;

        mtxQueueInput.lock();
        if (queueInput.empty()) {
            mtxQueueInput.unlock();
            if (bReading)
            {
                continue;
            } else {
                break;
            }
        } else {
            /* get an input frame from input frames queue */
            pairIndexImage = queueInput.front();  // queueの先頭からインデックスと画像のペアを取得
            queueInput.pop();  // queueポップ
            pairIndexImageYolo = queueInputYolo.front();  // queueの先頭からインデックスと画像のペアを取得
            queueInputYolo.pop();  // queueポップ
            mtxQueueInput.unlock();
        }

        vector<vector<float>> res;  // 結果格納用変数


        /* feed input frame into DPU Task with mean value */
        setInputImageForYOLO(task, pairIndexImageYolo.second, mean);

        /* invoke the running of DPU for YOLO-v3 */
        dpuRunTask(task);

        postProcessFPS(task, pairIndexImage.second, width, height, res);


        // メモリ解放
        free_image(pairIndexImageYolo.second);

        /* push the image into display frame queue */
        mtxQueueShow.lock();
        results_fps.push_back(res);  // グローバルなvectorへと追加
        results_fps_num.push_back(pairIndexImageYolo.first);  // グローバルなvectorへと追加
        queueFPS.push(pairIndexImage.first);  // 推論済画像のインデックスをqueueへ格納
        mtxQueueShow.unlock();
    }
}

/**
 * @brief Entry for running YOLO-v3-tiny neural network for AIエッジコンテスト
 *
 */
int main(const int argc, const char** argv) {

    if (argc <= 2) {
        cout << "Usage of this exe: ./yolo video_name[string] v"
             << endl;
        cout << "Usage of this exe: ./yolo image_name[string] i"
             << endl;
        cout << "Usage of this exe: ./yolo hoge c (for single-thread FPS & score check)"
             << endl;
        cout << "Usage of this exe: ./yolo hoge f num_begin num_end (for multi-thread FPS check)"
             << endl;
        cout << "Usage of this exe: ./yolo hoge t (for validation score check)"
             << endl;

        return -1;
    }

    string model = argv[2];

    if (argc == 5) {
        image_num_begin = atoi(argv[3]);  // マルチスレッド計測時にのみ考慮
        image_num_end = atoi(argv[4]);  // マルチスレッド計測時にのみ考慮
    }
    
    if (model == "v") {
  
        /* Attach to DPU driver and prepare for running */
        dpuOpen();
 
        /* Load DPU Kernels for YOLO-v3-tiny-contest network model */
        DPUKernel *kernel = dpuLoadKernel("yolo_tiny_contest");
        vector<DPUTask *> task(4);

        /* Create 4 DPU Tasks for YOLO-v3-tiny-contest network model */
        generate(task.begin(), task.end(),
        std::bind(dpuCreateTask, kernel, 0));

        /* Spawn 6 threads:
        - 1 thread for reading video frame
        - 4 identical threads for running network model
        - 1 thread for displaying frame in monitor
        */
        array<thread, 6> threadsList = {
            thread(readFrame, argv[1]),
            thread(displayFrame),
            thread(runYOLO_video, task[0]),
            thread(runYOLO_video, task[1]),
            thread(runYOLO_video, task[2]),
            thread(runYOLO_video, task[3]),
        };

        for (int i = 0; i < 6; i++) {
            threadsList[i].join();
        }

        /* Destroy DPU Tasks & free resources */
        for_each(task.begin(), task.end(), dpuDestroyTask);

        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;

    } else if (model == "i") {

        is_video=0;
        dpuOpen();
        Mat img;
        image img_yolo = load_image(img, argv[1]);
        DPUKernel *kernel = dpuLoadKernel("yolo_tiny_contest");
        DPUTask* task = dpuCreateTask(kernel, 0);

        runYOLOContest(task, img, img_yolo);
        imwrite("result.jpg", img);

        resize(img, img, Size(img.cols/4, img.rows/4));  // to smaller size
        namedWindow("Result", CV_WINDOW_AUTOSIZE);
        imshow("Result", img);
        waitKey(0);  // 何か押したら

        dpuDestroyTask(task);
         /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;

    } else if (model == "c") {  // シングルスレッドFPS計測 & スコア計算

        is_video = 0;
        dpuOpen();
        DPUKernel *kernel = dpuLoadKernel("yolo_tiny_contest");
        DPUTask* task = dpuCreateTask(kernel, 0);

        runYOLOContestCheck(task);

        dpuDestroyTask(task);
        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;

    } else if (model == "f") {  // マルチスレッドFPS計測

        /* Attach to DPU driver and prepare for running */
        dpuOpen();

        /* Load DPU Kernels for YOLO-v3-tiny-contest network model */
        DPUKernel *kernel = dpuLoadKernel("yolo_tiny_contest");
        vector<DPUTask *> task(4);

        /* Create 4 DPU Tasks for YOLO-v3-tiny-contest network model */
        generate(task.begin(), task.end(),
        std::bind(dpuCreateTask, kernel, 0));

        /* Spawn 6 threads:
        - 1 thread for reading video frame
        - 4 identical threads for running YOLO-v3 network model
        - 1 thread for displaying frame in monitor
        */
        array<thread, 6> threadsList = {
            thread(readFrameCheck),
            thread(displayFPS),
            thread(runYOLO_FPS, task[0]),
            thread(runYOLO_FPS, task[1]),
            thread(runYOLO_FPS, task[2]),
            thread(runYOLO_FPS, task[3]),
        };

        for (int i = 0; i < 6; i++) {
            threadsList[i].join();
        }

        /* Destroy DPU Tasks & free resources */
        for_each(task.begin(), task.end(), dpuDestroyTask);

        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;

    } else if (model == "t") {  // validationデータセットのスコア計算

        is_video = 0;
        dpuOpen();
        DPUKernel *kernel = dpuLoadKernel("yolo_tiny_contest");
        DPUTask* task = dpuCreateTask(kernel, 0);

        runYOLOValidationCheck(task);

        dpuDestroyTask(task);

         /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;

      } else {
        cout << "unknow type !" << endl;
        cout << "Usage of this exe: ./yolo video_name[string] v"
             << endl;
        cout << "Usage of this exe: ./yolo image_name[string] i"
             << endl;
        cout << "Usage of this exe: ./yolo hoge f num_begin num_end (for multi-thread FPS check)"
             << endl;
        cout << "Usage of this exe: ./yolo hoge f (for multi-thread FPS check)"
             << endl;
        cout << "Usage of this exe: ./yolo hoge t (for validation score check)"
             << endl;

        return -1;
    }
    
}
