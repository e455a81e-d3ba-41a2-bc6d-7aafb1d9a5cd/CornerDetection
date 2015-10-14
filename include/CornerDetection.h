#ifndef CORNER_DETECTION_H
#define CORNER_DETECTION_H

#include "Definitions.h"
#include <opencv2/core/core.hpp>

namespace iviso {

    //BORDER_TYPE is ignored by the current implementation and is therefore always BORDER_CONSTANT 0
    IVISO_API void cornerHarris(const cv::Mat& input, cv::Mat& output, int blockSize, int apertureSize, double k, int borderType = 0);
}

#endif //CORNER_DETECTION_H
