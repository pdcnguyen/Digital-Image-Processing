//============================================================================
// Name    : Dip3.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip3.h"

#include <stdexcept>

using namespace cv;
using namespace std;

namespace dip3
{

   const char *const filterModeNames[NUM_FILTER_MODES] = {
       "FM_SPATIAL_CONVOLUTION",
       "FM_FREQUENCY_CONVOLUTION",
       "FM_SEPERABLE_FILTER",
       "FM_INTEGRAL_IMAGE",
   };

   /**
    * @brief Generates 1D gaussian filter kernel of given size
    * @param kSize Kernel size (used to calculate standard deviation)
    * @returns The generated filter kernel
    */
   cv::Mat_<float> createGaussianKernel1D(int kSize)
   {

      Mat kernel = Mat::ones(1, kSize, CV_32FC1);
      int mean = kSize / 2;
      float sigma = kSize / 5.0;

      for (int i = 0; i < kSize; i++)
      {
         kernel.at<float>(0, i) = 1 / (2 * sigma * CV_PI) * exp(-0.5 * ((i - mean) * (i - mean) / (sigma * sigma)));
      }

      kernel *= (1 / sum(kernel)[0]);

      return kernel;
   }

   /**
    * @brief Generates 2D gaussian filter kernel of given size
    * @param kSize Kernel size (used to calculate standard deviation)
    * @returns The generated filter kernel
    */
   cv::Mat_<float> createGaussianKernel2D(int kSize)
   {

      Mat kernel = Mat::ones(kSize, kSize, CV_32FC1);
      int mean = kSize / 2;
      float sigma = kSize / 5.0;

      for (int i = 0; i < kSize; i++)
      {
         for (int j = 0; j < kSize; j++)
         {
            kernel.at<float>(i, j) = 1 / (2 * sigma * sigma * CV_PI) * exp(-0.5 * (((i - mean) * (i - mean) + (j - mean) * (j - mean)) / (sigma * sigma)));
         }
      }

      kernel *= (1 / sum(kernel)[0]);

      return kernel;
   }

   /**
    * @brief Performes a circular shift in (dx,dy) direction
    * @param in Input matrix
    * @param dx Shift in x-direction
    * @param dy Shift in y-direction
    * @returns Circular shifted matrix
    */
   cv::Mat_<float> circShift(const cv::Mat_<float> &in, int dx, int dy)
   {

      // 2 times % to get positive remainder number dx
      dx = ((dx % in.rows) + in.rows) % in.rows;
      // 2 times % to get positive remainder number dy
      dy = ((dy % in.cols) + in.cols) % in.cols;
      // new image dst
      cv::Mat dst = cv::Mat::zeros(in.size(), in.type());
      // transfer the row
      for (int i = 0; i < in.rows; i++)
      {
         int newrow = (i + dx) % in.rows;
         // transfer the col
         for (int j = 0; j < in.cols; j++)
         {
            int newcol = (j + dy) % in.cols;
            // circle shift
            dst.at<float>(newrow, newcol) = in.at<float>(i, j);
         }
      }
      // get the result image
      dst.copyTo(in);

      return in;
   }

   /**
    * @brief Performes convolution by multiplication in frequency domain
    * @param in Input image
    * @param kernel Filter kernel
    * @returns Output image
    */
   cv::Mat_<float> frequencyConvolution(const cv::Mat_<float> &in, const cv::Mat_<float> &kernel)
   {
      Mat mask = Mat::zeros(in.rows, in.cols, CV_64FC1);

      kernel.copyTo(mask(cv::Rect(0, 0, kernel.cols, kernel.rows)));

      int offsetRow = kernel.rows / 2;
      int offsetCol = kernel.cols / 2;

      mask = circShift(mask, -offsetRow, -offsetCol);

      Mat forwardImg;
      Mat forwardMask;
      Mat forwardFilteredImg;
      Mat backwardImg;

      dft(in, forwardImg, 0);
      dft(mask, forwardMask, 0);

      mulSpectrums(forwardImg, forwardMask, forwardFilteredImg, 0);

      dft(forwardFilteredImg, backwardImg, DFT_INVERSE + DFT_SCALE);

      return backwardImg;
   }

   /**
    * @brief  Performs UnSharp Masking to enhance fine image structures
    * @param in The input image
    * @param filterMode How convolution for smoothing operation is done
    * @param size Size of used smoothing kernel
    * @param thresh Minimal intensity difference to perform operation
    * @param scale Scaling of edge enhancement
    * @returns Enhanced image
    */
   cv::Mat_<float> usm(const cv::Mat_<float> &in, FilterMode filterMode, int size, float thresh, float scale)
   {
      Mat imgClone = in.clone();

      Mat smoothImg = smoothImage(in, size, filterMode);

      Mat subtractedImg = in - smoothImg;

      for (int i = 0; i < subtractedImg.rows; i++)
      {
         for (int j = 0; j < subtractedImg.cols; j++)
         {
            if (subtractedImg.at<float>(i, j) > thresh)
            {
               imgClone.at<float>(i, j) += subtractedImg.at<float>(i, j) * scale;
            }
         }
      }

      Mat finalImg;

      threshold(imgClone, finalImg, 255, 200, THRESH_TRUNC);

      return imgClone;
   }

   /**
    * @brief Convolution in spatial domain
    * @param src Input image
    * @param kernel Filter kernel
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
    * @brief Convolution in spatial domain by seperable filters
    * @param src Input image
    * @param size Size of filter kernel
    * @returns Convolution result
    */
   cv::Mat_<float> separableFilter(const cv::Mat_<float> &src, const cv::Mat_<float> &kernel)
   {
      Mat imgClone = spatialConvolution(src, kernel);
      transpose(imgClone, imgClone);

      imgClone = spatialConvolution(imgClone, kernel);
      transpose(imgClone, imgClone);

      return imgClone;
   }

   /**
    * @brief Convolution in spatial domain by integral images
    * @param src Input image
    * @param size Size of filter kernel
    * @returns Convolution result
    */
   cv::Mat_<float> satFilter(const cv::Mat_<float> &src, int size)
   {

      // optional

      return src;
   }

   /* *****************************
     GIVEN FUNCTIONS
   ***************************** */

   /**
    * @brief Performs a smoothing operation but allows the algorithm to be chosen
    * @param in Input image
    * @param size Size of filter kernel
    * @param type How is smoothing performed?
    * @returns Smoothed image
    */
   cv::Mat_<float> smoothImage(const cv::Mat_<float> &in, int size, FilterMode filterMode)
   {
      switch (filterMode)
      {
      case FM_SPATIAL_CONVOLUTION:
         return spatialConvolution(in, createGaussianKernel2D(size)); // 2D spatial convolution
      case FM_FREQUENCY_CONVOLUTION:
         return frequencyConvolution(in, createGaussianKernel2D(size)); // 2D convolution via multiplication in frequency domain
      case FM_SEPERABLE_FILTER:
         return separableFilter(in, createGaussianKernel1D(size)); // seperable filter
      case FM_INTEGRAL_IMAGE:
         return satFilter(in, size); // integral image
      default:
         throw std::runtime_error("Unhandled filter type!");
      }
   }

}
