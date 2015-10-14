#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "CornerDetection.h"

using namespace cv;

enum class Direction {
    X,
    Y
};

namespace {

    int sobel_x_kernel[3][3] =
        {{-1, 0, 1},
         {-2, 0, 2},
         {-1, 0, 1}};
    int sobel_y_kernel[3][3] =
        {{-1, -2, -1},
         {0,  0,  0},
         {1,  2,  1}};
    int box_kernel[3][3] = {{1, 1, 1}, {1, 1, 1}, {1 ,1 ,1}};
}

template<typename T>
T CheckBorderGetPixel(const Mat& src, int x, int y) {
    if ((x < 0) || (x >= src.cols) || (y < 0) || (y >= src.rows)) {
        return 0;
    } else {
        return src.at<T>(y, x);
    }
}


static void convoluteImage(const Mat& input, Mat& output, int ddepth, const Mat& kernel, double scale = 1)
{
    constexpr int size = 3 / 2;
    output.create(input.size(), ddepth);

    for (int x = 0; x < input.cols ; x++)
    {
        for (int y = 0; y < input.rows; y++) {
            double accumulator = 0;

            for (int kx = -size; kx <= size; kx++) {
                for (int ky = -size; ky <= size; ky++) {
                    accumulator += scale * CheckBorderGetPixel<float>(input, x + kx, y + ky) * kernel.at<int>(kx + size, ky + size);

                }
            }
            //accumulator = floorf(accumulator * 10000000) / 10000000;
            output.at<float>(y,x) = static_cast<float>(accumulator);
        }

    }
}

static void applyBoxFilter(const Mat& input, Mat& output, int ddepth)
{
    auto kernel = Mat(3,3, CV_32S, &box_kernel);

    //convoluteImage(input, output, ddepth, kernel, 1);
    filter2D(input, output, ddepth, kernel, Point(-1, -1), 0);
}

static void Sobel(const Mat& input, Mat& output, int ddepth, double scale, Direction dir) {

    auto kernel = Mat();
    if (dir == Direction::Y)
        kernel = Mat(3,3, CV_32S, &sobel_x_kernel);
    else
        kernel = Mat(3,3, CV_32S, &sobel_y_kernel);

    convoluteImage(input, output, ddepth, kernel, scale);

}

void eigenValues2x2(const Vec3f& input, Vec2f& output)
{
    const double& A = input[0];
    const double& B = input[1];
    const double& C = input[2];

    double u = (A + C)*0.5;
    double v = std::sqrt((A - C)*(A - C)*0.25 + B*B);
    double l1 = u + v;
    double l2 = u - v;
    output[0] = static_cast<float>(l1);
    output[1] = static_cast<float>(l2);
}

static void calcEigenVals( const cv::Mat& input, cv::Mat& output, int blockSize, int apertureSize)
{

    double scale = static_cast<double>((1 << ((apertureSize > 0 ? apertureSize : 3) - 1)) * blockSize);
    scale = 1.0/ scale;

    auto size = static_cast<Size>(input.size());

    auto Dx = Mat();
    auto Dy = Mat();


    Sobel(input, Dx, CV_32F, scale, Direction::X);
    Sobel(input, Dy, CV_32F, scale, Direction::Y);

    auto cov = Mat(size, CV_32FC3);
    for( int i = 0; i < size.height; i++ ) {
        for(int j = 0; j < size.width; j++ )
        {
            float dx = Dx.at<float>(i,j);
            float dy = Dy.at<float>(i,j);

            cov.at<Vec3f>(i,j)[0] = dx*dx;
            cov.at<Vec3f>(i,j)[1] = dx*dy;
            cov.at<Vec3f>(i,j)[2] = dy*dy;
        }
    }

    applyBoxFilter(cov, cov, cov.depth());

    //boxFilter(cov, cov, cov.depth(), Size(blockSize, blockSize), Point(-1,-1), false,  BORDER_DEFAULT);

    for (int j = 0; j < size.height; j++) {

        for (int i = 0; i < size.width; i++) {
            const auto& row = cov.at<Vec3f>(j, i);
            eigenValues2x2(row, output.at<Vec2f>(j, i));
        }
    }
}

void iviso::cornerHarris(const Mat &input, Mat &output, int blockSize, int apertureSize, double k, int borderType)
{

    //MatExpr is a "hidden" proxy class therefore explicit cast see Scott Meyers "Effective Modern C++" Item 6
    auto EigenVals = static_cast<Mat>(Mat::zeros(input.size(), CV_32FC(2)));
    Mat tmp;
    input.convertTo(tmp, CV_32F);
    output.create(input.size(), CV_32FC1);

    calcEigenVals(tmp, EigenVals, blockSize, apertureSize);

    for (int j = 0; j < input.rows; j++) {
        for (int i = 0; i < input.cols; i++) {
            auto lambda1 = EigenVals.at<Vec2f>(j,i)[0];
            auto lambda2 = EigenVals.at<Vec2f>(j,i)[1];
            output.at<float>(j,i) = lambda1*lambda2 - static_cast<float>(k)*pow((lambda1 + lambda2), 2.0f);
        }
    }
}


