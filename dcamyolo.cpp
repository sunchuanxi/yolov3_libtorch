#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <librealsense2/rs.hpp>

#include "Darknet.h"

int main(int argc, const char *argv[]) {
    if (argc != 1) {
        std::cerr << "usage: dcamyolo\n";
        return -1;
    }

    // Configuration
    int fps = 30;
    int width = 640;
    int height = 480;

    int class_num = 2;
    float confidence = 0.35;
    float nms_thresh = 0.45;
    int input_image_size = 416;

    // Find Torch device type
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

    // Initialize Darknet
    Darknet net("../models/yolov3.cfg", &device);

    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << std::endl;
    net.load_weights("../models/yolov3.weights");
    std::cout << "weight loaded ..." << std::endl << "transfering Darknet to device" << std::endl;

    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;

    // Initialize camera
    rs2::pipeline pipe;     // Contruct a pipeline which abstracts the device
    rs2::config cfg;        // Create a configuration for configuring the pipeline with a non default profile

    cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps); // Add desired streams to configuration
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

    pipe.start(cfg); // Instruct pipeline to start streaming with the requested configuration

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    std::cout << "dropping several first frames to let auto-exposure stabilize" << std::endl;
    rs2::frameset frames;
    for (int i = 0; i < 30; i++) frames = pipe.wait_for_frames(); // Wait for all configured streams to produce a frame

    // Variable Initializations
    cv::namedWindow("Yolo V3 in C++", 1);
    
    cv::Mat resized_image;
    int acc_frame_cnt = 0;
    int accumulated_duration = 0;
    int acc_fps = 0;

    // Starting pipeline
    std::cout << "starting pipeline ...." << std::endl;
    while (true) {
        // Read frames from camera into -> cv::Mat origin_image
        frames = pipe.wait_for_frames();
        
        rs2::frame color_frame = frames.get_color_frame(); // Get each frame
        rs2::frame depth_frame = frames.get_depth_frame();

        // Creating OpenCV matrix from IR image
        cv::Mat origin_image(cv::Size(width, height), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(width, height), CV_16UC1, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        acc_frame_cnt++;

        // Preprocess image (resize, put on GPU)
        cv::cvtColor(origin_image, resized_image, cv::COLOR_RGB2BGR);
        cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

        cv::Mat img_float;
        resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

        auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, input_image_size, input_image_size, 3});
        img_tensor = img_tensor.permute({0, 3, 1, 2});
        auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);
        
        // Yolo (Darknet + NMS)
        auto start = std::chrono::high_resolution_clock::now(); // Start measuring time

        auto output = net.forward(img_var);
        auto result = net.write_results(output, class_num, confidence, nms_thresh);
        
        auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        accumulated_duration += duration.count();

        // Display FPS after every few seconds
        if (accumulated_duration > 2000) {
            acc_fps = (int)(((float)(acc_frame_cnt) / (float)(accumulated_duration)) * 1000.0);
            std::cout << "average fps in last few seconds : " << acc_fps << endl;

            acc_frame_cnt = 0;
            accumulated_duration = 0;
        }

        // Extract Yolo boxes and display in window
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
                cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 128 * (1 - result_data[i][7]), 255 * result_data[i][7]), 2 + 2 * (1 - result_data[i][7]), 1, 0);
            }

            cv::imshow("Yolo V3 in C++", origin_image);
        }

        if (cv::waitKey(1) == 27) break; // Press escape to exit
    }
    std::cout << "Done" << std::endl;

    return 0;
}
