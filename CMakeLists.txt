cmake_minimum_required(VERSION 3.10)
project(RedDetection)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "C:/Users/Andreea/opencv/build" CACHE PATH "")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(RedDetection main.cpp)

target_link_libraries(RedDetection PRIVATE ${OpenCV_LIBS})
