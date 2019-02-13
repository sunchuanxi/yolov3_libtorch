#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unistd.h>

#include "Darknet.h"

using namespace std;
using namespace std::chrono;

int main(int argc, const char *argv[])
{
    if (argc != 1)
    {
        std::cerr << "usage: yolo-app\n";
        return -1;
    }

    int class_num = 2;
    float confidence = 0.35;
    float nms_thresh = 0.45;

    torch::DeviceType device_type;

    if (torch::cuda::is_available())
    {
        device_type = torch::kCUDA;
        std::cout << "Running on GPU" << std::endl;
    }
    else
    {
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

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) // check if we succeeded
        return -1;

    cv::namedWindow("Yolo V3 in C++", 1);

    while (true)
    {
        cap >> origin_image;

        cv::cvtColor(origin_image, resized_image, cv::COLOR_RGB2BGR);
        cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

        cv::Mat img_float;
        resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

        auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, input_image_size, input_image_size, 3});
        img_tensor = img_tensor.permute({0, 3, 1, 2});
        auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);

        // auto start = std::chrono::high_resolution_clock::now();

        auto output = net.forward(img_var);

        // filter result by NMS
        auto result = net.write_results(output, class_num, confidence, nms_thresh);

        // auto end = std::chrono::high_resolution_clock::now();

        // auto duration = duration_cast<milliseconds>(end - start);

        // It should be known that it takes longer time at first time
        // std::cout << "inference taken : " << duration.count() << " ms" << endl;

        if (result.dim() == 1)
        {
            // std::cout << "no object found" << endl;
        }
        else
        {
            int obj_num = result.size(0);

            // std::cout << obj_num << " objects found" << endl;

            float w_scale = float(origin_image.cols) / input_image_size;
            float h_scale = float(origin_image.rows) / input_image_size;

            result.select(1, 1).mul_(w_scale);
            result.select(1, 2).mul_(h_scale);
            result.select(1, 3).mul_(w_scale);
            result.select(1, 4).mul_(h_scale);

            auto result_data = result.accessor<float, 2>();

            for (int i = 0; i < result.size(0); i++)
            {
                // std::cout << result_data[i][7] << std::endl;
                cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 128 * (1 - result_data[i][7]), 255 * result_data[i][7]), 2 + 2 * (1 - result_data[i][7]), 1, 0);
            }

            cv::imshow("Yolo V3 in C++", origin_image);
        }

        cv::waitKey(30);
    }
    std::cout << "Done" << endl;

    return 0;
}
