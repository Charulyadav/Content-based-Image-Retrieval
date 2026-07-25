/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 2: Content-Based Image Retrieval

  function prototypes for image feature extraction.
  each feature function takes an image (cv::Mat) and writes a feature vector (std::vector<float>) suitable for storage in a CSV file.
*/

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

/*
  extracts a 7x7 BGR patch from the center of the image as a feature vector.
  output length = 7 * 7 * 3 = 147 floats, laid out as
  [B,G,R,B,G,R,...] in row-major order.

  arguments:
    src  - input BGR image (CV_8UC3)
    feat - output feature vector (cleared and resized internally)
  returns 0 on success, -1 if src is empty or smaller than 7x7.
*/
int computeBaseline(const cv::Mat &src, std::vector<float> &feat);

/*
  computes a 3D RGB color histogram of the entire image.
  the histogram is binned with `bins` divisions per channel (default 8),
  producing a feature vector of length bins^3. values are stored as raw
  counts; the distance metric should normalize before computing intersection.

  arguments:
    src  - input BGR image (CV_8UC3)
    feat - output flattened histogram, length = bins * bins * bins
    bins - bins per channel (defaults to 8 to match prof's reference)
  returns 0 on success, -1 if src is empty.
*/
int computeRGBHist(const cv::Mat &src, std::vector<float> &feat, int bins = 8);

/*
  computes two 3D RGB color histograms covering the top half and bottom
  half of the image, concatenated into a single feature vector.
  each histogram uses `bins` divisions per channel (default 8).
  total length = 2 * bins^3 (1024 floats with default bins).

  layout of output: [top_hist (bins^3 floats), bottom_hist (bins^3 floats)]

  arguments:
    src  - input BGR image (CV_8UC3)
    feat - output concatenated histograms
    bins - bins per channel (default 8)
  returns 0 on success, -1 if src is empty.
*/
int computeTopBottomHist(const cv::Mat &src, std::vector<float> &feat, int bins = 8);

/*
  computes a combined color+texture feature vector.
  color part: 8-bin RGB histogram of the whole image (bins^3 floats).
  texture part: 1D histogram of Sobel gradient magnitudes (textureBins floats).
  layout: [color_hist (bins^3), texture_hist (textureBins)]
  total length = bins^3 + textureBins.

  arguments:
    src         - input BGR image (CV_8UC3)
    feat        - output concatenated feature vector
    bins        - color bins per channel (default 8)
    textureBins - bins for the gradient magnitude histogram (default 16)
  returns 0 on success, -1 if src is empty.
*/
int computeColorTexture(const cv::Mat &src, std::vector<float> &feat, int bins = 8, int textureBins = 16);

/*
  custom feature for retrieving green dumpsters/containers.
  layout: [greenRatio (1), colorTexture (bins^3 + textureBins)]
  greenRatio = fraction of pixels classified as "saturated green" in HSV space.
  combines with a Color+Texture histogram for color distribution and edge density.

  arguments:
    src         - input BGR image (CV_8UC3)
    feat        - output feature vector
    bins        - color histogram bins per channel (default 8)
    textureBins - texture histogram bins (default 16)
  returns 0 on success, -1 if src is empty.
*/
int computeCustomDumpster(const cv::Mat &src, std::vector<float> &feat, int bins = 8, int textureBins = 16);

#endif