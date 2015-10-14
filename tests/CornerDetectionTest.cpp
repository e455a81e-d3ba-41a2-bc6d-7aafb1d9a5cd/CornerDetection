#include "gtest/gtest.h"
#include "CornerDetection.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

TEST(test1, sort_test)
{

    float b[4][4] = {
        { 12, 0, 0, 0},
        {  0, 5, 1, 0},
        {  0, 0, 1, 2},
        {  0, 0, 2, 1}
    };

    cv::Mat E;
    cv::Mat M(4,4,CV_32FC1,b);
    cv::cornerMinEigenVal(M,E, 3,3);

    std::cout << E << std::endl;
    cv::cornerHarris(M, E, 3, 3, 0.04);
    std::cout << E << std::endl;

}

TEST(test2, myEigen) {

    float b[4][4] = {
        { 12, 0, 0, 0},
        {  0, 5, 1, 0},
        {  0, 0, 1, 2},
        {  0, 0, 2, 1}
    };

    cv::Mat E(4,4, CV_32FC(6));
    cv::Mat M(4,4,CV_32FC1,b);
    cv::cornerEigenValsAndVecs(M,E, 3,3);

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            std::cout << E.at<cv::Vec6f>(j, i)[0] <<
            "\t" << E.at<cv::Vec6f>(j, i)[1] << std::endl;
        }
    }
}
