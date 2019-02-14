#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unistd.h>
#include <stdio.h>

#include "Darknet.h"

using namespace std;
using namespace std::chrono;

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: yolo-app <cam_id>\n";
        return -1;
    }

    int cam_id = atoi(argv[1]);

    int class_num = 2;
    float confidence = 0.35;
    float nms_thresh = 0.45;

    torch::DeviceType device_type;

    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "Running on GPU" << std::endl;
    }
    else {
        device_type = torch::kCPU;
        std::cout << "Running on CPU" << std::endl;
    }
    torch::Device device(device_type);

    // input image size for YOLO v3
    int input_image_size = 416;

    Darknet net("../models/yolov3.cfg", &device);

    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << endl;
    net.load_weights("../models/yolov3.weights");
    std::cout << "weight loaded ..." << endl;

    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;

    cv::Mat origin_image, resized_image;

    cv::VideoCapture cap(cam_id);

    if (!cap.isOpened()) // check if we succeeded
        return -1;

    cv::namedWindow("Yolo V3 in C++", 1);

    int acc_frame_cnt = 0;
    int accumulated_duration = 0;
    int acc_fps = 0;

    while (true) {
        cap >> origin_image;
        acc_frame_cnt++;

        cv::cvtColor(origin_image, resized_image, cv::COLOR_RGB2BGR);
        cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

        cv::Mat img_float;
        resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

        auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, input_image_size, input_image_size, 3});
        img_tensor = img_tensor.permute({0, 3, 1, 2});
        auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);

        auto start = std::chrono::high_resolution_clock::now();

        auto output = net.forward(img_var);

        // filter result by NMS
        auto result = net.write_results(output, class_num, confidence, nms_thresh);

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start);
        accumulated_duration += duration.count();

        if (accumulated_duration % 5000 == 0) {
            acc_fps = (int)(((float)(acc_frame_cnt) / (float)(accumulated_duration)) * 1000.0);
            std::cout << "average fps in last 5 seconds : " << acc_fps << endl;

            acc_frame_cnt = 0;
            accumulated_duration = 0;
        }

        if (result.dim() != 1) {
            int obj_num = result.size(0);

            float w_scale = float(origin_image.cols) / input_image_size;
            float h_scale = float(origin_image.rows) / input_image_size;

            result.select(1, 1).mul_(w_scale);
            result.select(1, 2).mul_(h_scale);
            result.select(1, 3).mul_(w_scale);
            result.select(1, 4).mul_(h_scale);

            auto result_data = result.accessor<float, 2>();

            for (int i = 0; i < result.size(0); i++) {
                // std::cout << result_data[i][7] << std::endl;
                cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 128 * (1 - result_data[i][7]), 255 * result_data[i][7]), 2 + 2 * (1 - result_data[i][7]), 1, 0);
            }

            cv::imshow("Yolo V3 in C++", origin_image);
        }

        if (cv::waitKey(1) == 27) break;
    }
    std::cout << "Done" << endl;

    return 0;
}
