#include <opencv2/imgproc/imgproc.hpp>
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
}

template<typename T>
T CheckBorderGetPixel(const Mat& src, int x, int y) {
    if ((x < 0) || (x >= src.cols) || (y < 0) || (y >= src.rows)) {
        return 0;
    } else {
        return src.at<T>(y, x);
    }
}


static void Sobel(const Mat& input, Mat& output, int ddepth, double scale, Direction dir)
{
    constexpr int size = 3 / 2;
    output.create(input.size(), ddepth);
    auto kernel = Mat();

    if (dir == Direction::Y)
        kernel = Mat(3,3, CV_32S, &sobel_x_kernel);
    else
        kernel = Mat(3,3, CV_32S, &sobel_y_kernel);


    for (int x = 0; x < input.cols ; x++)
    {
        for (int y = 0; y < input.rows; y++) {
            float accumulator = 0;

            for (int kx = -size; kx <= size; kx++) {
                for (int ky = -size; ky <= size; ky++) {
                    accumulator += scale * CheckBorderGetPixel<float>(input, x + kx, y + ky) * kernel.at<int>(kx + size, ky + size);

                }

            }
            //accumulator = floorf(accumulator * 10000000) / 10000000;
            output.at<float>(y,x) = accumulator;
        }

    }
}

static void calcEigenVals( const cv::Mat& input, cv::Mat& output, int blockSize, int apertureSize)
{
    double scale = (double)(1 << ((apertureSize > 0 ? apertureSize : 3) - 1)) * blockSize;
    scale = 1.0/ scale;

    auto size = static_cast<Size>(input.size());

    auto Dx = Mat();
    auto Dy = Mat();
    Sobel(input, Dx, CV_32F, scale, Direction::X);
    Sobel(input, Dy, CV_32F, scale, Direction::Y);

    for (int j = 0; j < size.height; j++) {
        for (int i = 0; i < size.width; i++) {

        }
    }


}

void iviso::cornerHarris(const Mat &input, Mat &output, int blockSize, int apertureSize, double k, int borderType)
{

    //MatExpr is a "hidden" proxy class therefore explicit cast see Scott Meyers "Effective Modern C++" Item 6
    auto EigenVals = static_cast<Mat>(Mat::zeros(input.size(), CV_32FC(6)));
    output.create(input.size(), CV_32FC1);

    cornerEigenValsAndVecs(input, EigenVals, blockSize, apertureSize, BORDER_DEFAULT);

    for (int j = 0; j < input.rows; j++) {
        for (int i = 0; i < input.cols; i++) {
            auto lambda1 = EigenVals.at<Vec6f>(j,i)[0];
            auto lambda2 = EigenVals.at<Vec6f>(j,i)[1];
            //High values of k may not fit into float type.
            output.at<float>(j,i) = lambda1*lambda2 - static_cast<float>(k)*pow((lambda1 + lambda2), 2.0f);
        }
    }
}

static void eigen2x2( const float* cov, float* dst, int n )
{
    for( int j = 0; j < n; j++ )
    {
        double a = cov[j*3];
        double b = cov[j*3+1];
        double c = cov[j*3+2];

        double u = (a + c)*0.5;
        double v = std::sqrt((a - c)*(a - c)*0.25 + b*b);
        double l1 = u + v;
        double l2 = u - v;

        double x = b;
        double y = l1 - a;
        double e = fabs(x);

        if( e + fabs(y) < 1e-4 )
        {
            y = b;
            x = l1 - c;
            e = fabs(x);
            if( e + fabs(y) < 1e-4 )
            {
                e = 1./(e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }

        double d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
        dst[6*j] = (float)l1;
        dst[6*j + 2] = (float)(x*d);
        dst[6*j + 3] = (float)(y*d);

        x = b;
        y = l2 - a;
        e = fabs(x);

        if( e + fabs(y) < 1e-4 )
        {
            y = b;
            x = l2 - c;
            e = fabs(x);
            if( e + fabs(y) < 1e-4 )
            {
                e = 1./(e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }

        d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
        dst[6*j + 1] = (float)l2;
        dst[6*j + 4] = (float)(x*d);
        dst[6*j + 5] = (float)(y*d);
    }
}


static void calcEigenValsVecs( const cv::Mat& _cov, cv::Mat& _dst )
{
    cv::Size size = _cov.size();
    if( _cov.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( int i = 0; i < size.height; i++ )
    {
        const float* cov = _cov.ptr<float>(i);
        float* dst = _dst.ptr<float>(i);

        eigen2x2(cov, dst, size.width);
    }
}
