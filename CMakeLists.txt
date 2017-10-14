cmake_minimum_required(VERSION 3.7)
project(quiz_gnb)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp classifier.cpp classifier.h)
add_executable(quiz_gnb ${SOURCE_FILES})