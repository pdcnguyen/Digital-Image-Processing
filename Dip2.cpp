//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip2.h"

#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

namespace dip2
{

    /**
     * @brief Convolution in spatial domain.
     * @details Performs spatial convolution of image and filter kernel.
     * @params src Input image
     * @params kernel Filter kernel
     * @returns Convolution result
     */
    cv::Mat_<float> spatialConvolution(const cv::Mat_<float> &src, const cv::Mat_<float> &kernel)
    {
        int offsetRow = kernel.rows / 2;
        int offsetCol = kernel.cols / 2;

        Mat flippedKernel;
        flip(kernel, flippedKernel, 0);
        flip(kernel, flippedKernel, 1);

        Mat expandedImg;
        cv::copyMakeBorder(src, expandedImg, offsetRow, offsetRow, offsetCol, offsetCol, BORDER_REPLICATE, 0);

        Mat imgClone = src.clone();

        for (int i = offsetRow; i < expandedImg.rows - offsetRow; i++)
        {
            for (int j = offsetCol; j < expandedImg.cols - offsetCol; j++)
            {

                Mat expandedImgFraction = expandedImg.rowRange(i - offsetRow, i + offsetRow + 1).colRange(j - offsetCol, j + offsetCol + 1).clone();

                expandedImgFraction = expandedImgFraction.mul(flippedKernel);

                imgClone.at<float>(i - offsetRow, j - offsetCol) = sum(expandedImgFraction)[0];
            }
        }

        return imgClone;
    }

    /**
     * @brief Moving average filter (aka box filter)
     * @note: you might want to use Dip2::spatialConvolution(...) within this function
     * @param src Input image
     * @param kSize Window size used by local average
     * @returns Filtered image
     */
    cv::Mat_<float> averageFilter(const cv::Mat_<float> &src, int kSize)
    {

        Mat kernel = Mat::ones(kSize, kSize, CV_32F) / (kSize * kSize);

        return spatialConvolution(src, kernel);
    }

    /**
     * @brief Median filter
     * @param src Input image
     * @param kSize Window size used by median operation
     * @returns Filtered image
     */
    cv::Mat_<float> medianFilter(const cv::Mat_<float> &src, int kSize)
    {
        int offset = kSize / 2;

        Mat expandedImg;
        cv::copyMakeBorder(src, expandedImg, offset, offset, offset, offset, BORDER_REPLICATE, 0);

        Mat imgClone = src.clone();

        for (int i = offset; i < expandedImg.rows - offset; i++)
        {
            for (int j = offset; j < expandedImg.cols - offset; j++)
            {
                Mat expandedImgFraction = expandedImg.rowRange(i - offset, i + offset + 1).colRange(j - offset, j + offset + 1).clone();

                cv::Mat flat = expandedImgFraction.reshape(1, expandedImgFraction.total() * expandedImgFraction.channels());
                std::vector<float> vec = expandedImgFraction.isContinuous() ? flat : flat.clone();
                std::sort(vec.begin(), vec.end());

                imgClone.at<float>(i - offset, j - offset) = vec[kSize * kSize / 2];
            }
        }

        return imgClone;
    }

    /**
     * @brief Bilateral filer
     * @param src Input image
     * @param kSize Size of the kernel
     * @param sigma_spatial Standard-deviation of the spatial kernel
     * @param sigma_radiometric Standard-deviation of the radiometric kernel
     * @returns Filtered image
     */
    cv::Mat_<float> bilateralFilter(const cv::Mat_<float> &src, int kSize, float sigma_spatial, float sigma_radiometric)
    {
        int offset = kSize / 2;

        Mat expandedImg;
        cv::copyMakeBorder(src, expandedImg, offset, offset, offset, offset, BORDER_REPLICATE, 0);

        Mat spatialKernel = Mat::ones(kSize, kSize, CV_32FC1);
        Mat radioKernel = Mat::ones(kSize, kSize, CV_32FC1);
        Mat combinedKernel = Mat::ones(kSize, kSize, CV_32FC1);

        Mat imgClone = src.clone();

        for (int i = offset; i < expandedImg.rows - offset; i++)
        {
            for (int j = offset; j < expandedImg.cols - offset; j++)
            {
                Mat expandedImgFraction = expandedImg.rowRange(i - offset, i + offset + 1).colRange(j - offset, j + offset + 1).clone();

                for (int ii = 0; ii < kSize; ii++)
                { // your class made me do this!!!!!!!!!!!!!!!!
                    for (int jj = 0; jj < kSize; jj++)
                    {
                        spatialKernel.at<float>(ii, jj) = exp(-1 * (pow((ii - offset), 2) + pow((jj - offset), 2))/(2*pow(sigma_spatial,2))); // Look at this abomination!!!
                        radioKernel.at<float>(ii, jj) = exp(-1 * pow(expandedImgFraction.at<float>(ii, jj) - expandedImgFraction.at<float>(offset, offset), 2)/(2*pow(sigma_radiometric,2)));//How can you sleep at night??
                    }
                }

                combinedKernel = spatialKernel.mul(radioKernel);

                expandedImgFraction = expandedImgFraction.mul(combinedKernel);

                imgClone.at<float>(i - offset, j - offset) = sum(expandedImgFraction)[0]/sum(combinedKernel)[0];
            }
        }

        return imgClone;
    }

    /**
     * @brief Non-local means filter
     * @note: This one is optional!
     * @param src Input image
     * @param searchSize Size of search region
     * @param sigma Optional parameter for weighting function
     * @returns Filtered image
     */
    cv::Mat_<float> nlmFilter(const cv::Mat_<float> &src, int searchSize, double sigma)
    {
        return src.clone();
    }

    /**
     * @brief Chooses the right algorithm for the given noise type
     * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
     */
    NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType)
    {
		if (noiseType == NOISE_TYPE_1) {
			return NR_MEDIAN_FILTER;
		} else if (noiseType == NOISE_TYPE_2) {
			return NR_BILATERAL_FILTER;
		}
    }

    cv::Mat_<float> denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType, dip2::NoiseReductionAlgorithm noiseReductionAlgorithm)
    {
		switch (noiseReductionAlgorithm) {
			case dip2::NR_MOVING_AVERAGE_FILTER:
				switch (noiseType) {
					case NOISE_TYPE_1:
						return dip2::averageFilter(src, 5);
					case NOISE_TYPE_2:
						return dip2::averageFilter(src, 7);
					default:
						throw std::runtime_error("Unhandled noise type!");
				}
			case dip2::NR_MEDIAN_FILTER:
				switch (noiseType) {
					case NOISE_TYPE_1:
						return dip2::medianFilter(src, 5);
					case NOISE_TYPE_2:
						return dip2::medianFilter(src, 5);
					default:
						throw std::runtime_error("Unhandled noise type!");
				}
			case dip2::NR_BILATERAL_FILTER:
				switch (noiseType) {
					case NOISE_TYPE_1:
						return dip2::bilateralFilter(src, 7, 400.0f, 400.0f);
					case NOISE_TYPE_2:
						return dip2::bilateralFilter(src, 5, 400.0f, 400.0f);
					default:
						throw std::runtime_error("Unhandled noise type!");
				}
			default:
				throw std::runtime_error("Unhandled filter type!");
		}
    }

    // Helpers, don't mind these

    const char *noiseTypeNames[NUM_NOISE_TYPES] = {
        "NOISE_TYPE_1",
        "NOISE_TYPE_2",
    };

    const char *noiseReductionAlgorithmNames[NUM_FILTERS] = {
        "NR_MOVING_AVERAGE_FILTER",
        "NR_MEDIAN_FILTER",
        "NR_BILATERAL_FILTER",
    };

}
