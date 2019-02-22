// include C++ libraries
#include <vector>
#include <pthread.h>

// include the librealsense C++ header file
#include <librealsense2/rs.hpp>

// include OpenCV header file
#include <opencv2/opencv.hpp>

class DepthCamArgs {
    public:
        DepthCamArgs(int i_th_id, int i_fps, int i_width, int i_height, std::vector<std::string> i_dev_srn);
        int thread_id;
        int fps;
        int width;
        int height;
        std::vector<std::string> device_serial_nums;
};

DepthCamArgs::DepthCamArgs(int i_th_id, int i_fps, int i_width, int i_height, std::vector<std::string> i_dev_srns) {
    this->thread_id = i_th_id;
    this->device_serial_nums = i_dev_srns;
    this->fps = i_fps;
    this->width = i_width;
    this->height = i_height;
}

void *RunDepthCam(void *threadarg) {
    // Read arguments
    DepthCamArgs *args = static_cast<DepthCamArgs *>(threadarg);
    int th_id = args->thread_id;
    int fps = args->fps;
    int width = args->width;
    int height = args->height;
    std::vector<std::string> device_sr_nums = args->device_serial_nums;

    std::vector<rs2::pipeline> vec_pipes;
    for (std::string dev_sr_nm : device_sr_nums) {
        // Contruct a pipeline which abstracts the device
        rs2::pipeline pipe;

        // Create a configuration for configuring the pipeline with a non default profile
        rs2::config cfg;

        // Add desired streams to configuration
        cfg.enable_device(dev_sr_nm);
        cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
        // cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

        // Instruct pipeline to start streaming with the requested configuration
        pipe.start(cfg);

        vec_pipes.push_back(pipe);
    }

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    rs2::frameset frames;
    for (int i = 0; i < 30; i++) { // Wait for all configured streams to produce a frame
        for (rs2::pipeline pipe : vec_pipes)
            frames = pipe.wait_for_frames();
    }

    // Create OpenCV windows
    cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Depth Image", cv::WINDOW_AUTOSIZE);
    
    // Poll each camera for cnt times
    cv::Mat all_color;
    // cv::Mat all_depth;
    std::vector<cv::Mat> vec_color;
    // std::vector<cv::Mat> vec_depth;
    int cnt = 0;
    while (cnt < 150)
    {
        cnt++;
        vec_color.clear();
        // vec_depth.clear();

        // Poll cameras sequentially (tried in different threads, it doesn't work apparently)
        for (rs2::pipeline pipe : vec_pipes) {
            rs2::frameset frames;
            frames = pipe.wait_for_frames();

            // Get each frame
            rs2::frame color_frame = frames.get_color_frame();
            // rs2::frame depth_frame = frames.get_depth_frame();

            // Creating OpenCV matrix from IR image
            cv::Mat color(cv::Size(width, height), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
            // cv::Mat depth(cv::Size(width, height), CV_16UC1, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);

            vec_color.push_back(color);
            // vec_depth.push_back(depth);
        }

        // Concatenate frames for display
        cv::hconcat(vec_color, all_color);
        // cv::hconcat(vec_depth, all_depth);

        // Display the image in GUI
        cv::imshow("Color Image", all_color);
        // cv::imshow("Depth Image", all_depth);

        if (th_id==0) if (cv::waitKey(1) == 27) break;
    }

    cv::destroyAllWindows();
}

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
    
    if (devices.size() < 1) {
        std::cout << "No cameras connected. Exiting." << std::endl;
        return 0;
    } else {
        std::cout << "Number of cameras: " << devices.size() << std::endl;
    }

    // Initialize threads for cameras
    pthread_t cams_thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    int rc;
    void *status;

    DepthCamArgs dcam_args = DepthCamArgs(0, fps, width, height, devs_sr);

    rc = pthread_create(&cams_thread, &attr, RunDepthCam, (void *) &dcam_args);
    if (rc){
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
    }

    rc = pthread_join(cams_thread, &status);
    if (rc) {
        printf("ERROR; return code from pthread_join() is %d\n", rc);
        exit(-1);
    }

    pthread_attr_destroy(&attr);

    return 0;
}
