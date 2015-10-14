//
// Created by leonhardt on 14.10.15.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <CornerDetection.h>

namespace {
    const char* WINDOW_NAME = "Scratchpad";
    const char* TEST_IMG_DIR = "TestData/checker.jpg";
}



#define tolerance 0.1e-20
double square(double i)
{
    return pow(i, 2.0);
}

double max(double a, double b)
{
    return a > b ? a : b;
}


void solve_eig(double A, double B, double C, double D,
               double* lambda1, double *v1x, double*v1y,
               double* lambda2, double *v2x, double*v2y )
{
    if(B*C <= tolerance  ) {
        *lambda1 = A;
        *lambda2 = D;
        return;
    }

    double tr = A + D;
    double det = A * D - B * C;
    double S = sqrt( square(tr/2.0) - det );
    *lambda1 = tr/2.0 + S;
    *lambda2 = tr/2.0 - S;
}


int main(int argc, char **argv)
{

    float f[2][2] = {{2,7},{-1,-6}};
    float b[4][4] = {
        { 12, 0, 0, 0},
        {  0, 5, 1, 0},
        {  0, 0, 1, 2},
        {  0, 0, 2, 1}
    };

    auto M = cv::Mat(4,4,CV_32FC1,b);
    auto E = cv::Mat(M.size(), CV_32FC1);

    double scale = (double)(1 << ((3 > 0 ? 3 : 3) - 1)) * 3;
    scale = 1.0/ scale;
    Sobel( M, E, CV_32F, 0, 1, 3, scale, 0, cv::BORDER_CONSTANT);
    std::cout << E << std::endl;

    iviso::cornerHarris(M,E,3,3,0.04);

    std::cout << E;
    /*
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            std::cout << E.at<cv::Vec6f>(j, i)[0] <<
            "\t" << E.at<cv::Vec6f>(j, i)[1];
        }
        std::cout <<  std::endl;
    }

     iviso::cornerHarris(M,E,3,3,0.04);
     std::cout << E << std::endl;

     cv::cornerHarris(M, E, 3, 3, 0.04);
     std::cout << E << std::endl;
     return 0;
     /*double lambda1 = 0;
     double v1x = 0;
     double v1y = 0;

     double lambda2 = 0;
     double v2x = 0;
     double v2y = 0;
     solve_eig(2,7,-1, -6, &lambda1, &v1x, &v1y, &lambda2, &v2x, &v2y);


     std::cout << lambda1 << "\t" << lambda2 << std::endl;
     return 0;
     float b[4][4] = {
         { 12, 0, 0, 0},
         {  0, 5, 1, 0},
         {  0, 0, 1, 2},
         {  0, 0, 2, 1}
     };

     cv::Mat E(4,4, CV_32FC(6));
     cv::Mat M(4,4,CV_32FC1,b);
     cv::cornerEigenValsAndVecs(M,E, 3,3);

 */
}

int test() {

    auto image = cv::imread(TEST_IMG_DIR, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat dst, dstNorm, dstNormScaled;
    dst = cv::Mat::zeros(image.size(), CV_32FC1);
    cv::cornerHarris(image, dst, 7, 5, 0.05);
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs( dstNorm, dstNormScaled );
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);


    for (int j = 0; j < dstNormScaled.rows; j++ ) {
        for (int i = 0; i < dstNormScaled.cols; i++) {
            if (static_cast<int>(dstNormScaled.at<float>(j,i)) > 200) {
                //dstNormScaled.at<float>(j,i) = 0;
                cv::circle(dstNormScaled, cv::Point(i,j), 5, cv::Scalar(0),2, 8, 0);
            }
        }
    }

    cv::imshow(WINDOW_NAME, dstNormScaled);
    cv::waitKey(0);
    return 0;
}