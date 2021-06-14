#CMakeLists.txt模板


cmake_minimum_required(VERSION 2.8)


#项目名
project(scanner)

set(CMAKE_BUILD_TYPE "Release")  
add_definitions("-DENABLE_SSE") 
set(CMAKE_CXX_STANDARD 14)    #使用Sophus库必须加这一句 （Sophus库只支持c++14）


#OpenCV库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(scanner my_scanner.cpp)

target_link_libraries(scanner ${OpenCV_LIBS})

