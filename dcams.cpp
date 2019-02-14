// include the librealsense C++ header file
#include <librealsense2/rs.hpp>

// include OpenCV header file
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    // Initializations
    int fps = 30;
    int width = 640;
    int height = 480;

    // Contruct a pipeline which abstracts the device
    rs2::pipeline pipe;

    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    // Add desired streams to configuration
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
    cfg.enable_stream(RS2_STREAM_INFRARED, width, height, RS2_FORMAT_Y8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

    // Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    rs2::frameset frames;
    for (int i = 0; i < 30; i++) frames = pipe.wait_for_frames(); // Wait for all configured streams to produce a frame

    namedWindow("Color Image", WINDOW_AUTOSIZE);
    namedWindow("IR Image", WINDOW_AUTOSIZE);
    namedWindow("Depth Image", WINDOW_AUTOSIZE);

    while (true)
    {
        frames = pipe.wait_for_frames();

        // Get each frame
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
        rs2::frame depth_frame = frames.get_depth_frame();

        // Creating OpenCV matrix from IR image
        Mat ir(Size(width, height), CV_8UC1, (void *)ir_frame.get_data(), Mat::AUTO_STEP);
        Mat color(Size(width, height), CV_8UC3, (void *)color_frame.get_data(), Mat::AUTO_STEP);
        Mat depth(Size(width, height), CV_16UC1, (void *)depth_frame.get_data(), Mat::AUTO_STEP);

        // Apply Histogram Equalization
        equalizeHist(ir, ir);
        applyColorMap(ir, ir, COLORMAP_JET);

        // Display the image in GUI
        imshow("Color Image", color);
        imshow("IR Image", ir);
        imshow("Depth Image", depth);

        if (waitKey(1) == 27) break;
    }

    return 0;
}
