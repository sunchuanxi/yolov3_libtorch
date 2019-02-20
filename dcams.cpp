// include C++ libraries
#include<vector>

// include the librealsense C++ header file
#include <librealsense2/rs.hpp>

// include OpenCV header file
#include <opencv2/opencv.hpp>

int main()
{
    // Initializations
    int fps = 30;
    int width = 640;
    int height = 480;

    // Get a list of cameras
    std::vector<std::string> devs_sr;
    rs2::context ctx = rs2::context();
    rs2::device_list devices = ctx.query_devices();
    for (rs2::device device : devices) {
        std::string dev_info = device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER);
        devs_sr.push_back(dev_info);
    }
    std::cout << "Number of cameras: " << devices.size() << std::endl;
    for (std::string str : devs_sr) std::cout << "device_id: " << str << std::endl;

    // Contruct a pipeline which abstracts the device
    rs2::pipeline pipe;

    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    // Add desired streams to configuration
    cfg.enable_device(devs_sr[0]);
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
    cfg.enable_stream(RS2_STREAM_INFRARED, width, height, RS2_FORMAT_Y8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

    // Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    rs2::frameset frames;
    for (int i = 0; i < 30; i++) frames = pipe.wait_for_frames(); // Wait for all configured streams to produce a frame

    cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("IR Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Depth Image", cv::WINDOW_AUTOSIZE);

    while (true)
    {
        frames = pipe.wait_for_frames();

        // Get each frame
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
        rs2::frame depth_frame = frames.get_depth_frame();

        // Creating OpenCV matrix from IR image
        cv::Mat ir(cv::Size(width, height), CV_8UC1, (void *)ir_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat color(cv::Size(width, height), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(width, height), CV_16UC1, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        // Apply Histogram Equalization
        equalizeHist(ir, ir);
        applyColorMap(ir, ir, cv::COLORMAP_JET);

        // Display the image in GUI
        cv::imshow("Color Image", color);
        cv::imshow("IR Image", ir);
        cv::imshow("Depth Image", depth);

        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}
