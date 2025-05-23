cmake_minimum_required(VERSION 3.10)
project(RedDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCV_DIR "C:/Users/Andreea/opencv/build" CACHE PATH "Path to OpenCV build directory")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(RedDetection main.cpp)
target_link_libraries(RedDetection PRIVATE ${OpenCV_LIBS})
