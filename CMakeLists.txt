cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolo-app)

find_package(realsense2 REQUIRED)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Torch REQUIRED)

aux_source_directory(. DIR_SRCS)

add_executable(yolo-cam yolo.cpp Darknet.cpp)
target_link_libraries(yolo-cam "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET yolo-cam PROPERTY CXX_STANDARD 11)

add_executable(dcams dcams.cpp)
include_directories(include ${realsense_INCLUDE_DIR})
target_include_directories(dcams PRIVATE ${realsense_INCLUDE_DIR})
target_link_libraries(dcams ${realsense2_LIBRARY} "${OpenCV_LIBS}")
set_property(TARGET dcams PROPERTY CXX_STANDARD 11)

add_executable(dcamyolo dcamyolo.cpp Darknet.cpp)
include_directories(include ${realsense_INCLUDE_DIR})
target_include_directories(dcamyolo PRIVATE ${realsense_INCLUDE_DIR})
target_link_libraries(dcamyolo ${realsense2_LIBRARY} "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET dcamyolo PROPERTY CXX_STANDARD 11)
