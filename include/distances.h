/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 2: Content-Based Image Retrieval

  function prototypes for distance metrics between feature vectors.
  all distances return a non-negative value; smaller = more similar.
*/

#ifndef DISTANCES_H
#define DISTANCES_H

#include <vector>

/*
  sum-of-squared-differences between two equal-length feature vectors.
  d(a, b) = sum_i (a[i] - b[i])^2

  arguments:
    a, b - feature vectors (must be the same length)
  returns the SSD as a float, or -1 if vectors have mismatched lengths.
*/
float ssd(const std::vector<float> &a, const std::vector<float> &b);

/*
  histogram intersection distance.
  both histograms are normalized to sum to 1.0, then the intersection
  similarity is min(a_i, b_i) summed across bins. the result is in [0, 1],
  where 1 means identical histograms and 0 means no overlap.
  we return DISTANCE, not similarity, so return value = 1 - intersection.
  lower = more similar (consistent with SSD).

  arguments:
    a, b - feature vectors (histograms, must be the same length)
  returns the distance as a float in [0, 1], or -1 if lengths mismatch.
*/
float histIntersection(const std::vector<float> &a, const std::vector<float> &b);

/*
  distance metric for the top+bottom concatenated histogram feature.
  splits each input vector into two equal halves (top half + bottom half),
  computes histogram intersection on each separately, and averages
  the two distances with equal weights.

  arguments:
    a, b - feature vectors (must be the same length and even-sized)
  returns the averaged distance in [0, 1], or -1 if lengths mismatch.
*/
float histIntersectionMulti(const std::vector<float> &a, const std::vector<float> &b);

/*
  distance metric for the color+texture feature vector.
  splits the input into a color histogram (first bins^3 entries) and a
  texture histogram (remaining entries), runs histogram intersection on
  each, and returns the equally-weighted average.

  arguments:
    a, b - feature vectors (must be the same length)
  returns the averaged distance, or -1 if lengths mismatch.
*/
float colorTextureDistance(const std::vector<float> &a, const std::vector<float> &b, int colorSize);

/*
  cosine distance between two equal-length feature vectors.
  d(a, b) = 1 - (a . b) / (||a|| * ||b||)
  works well for high-dimensional embeddings where SSD overemphasizes
  magnitude differences. Result is in [0, 2]:
  0 means identical direction, 1 means orthogonal, 2 means opposite.

  arguments:
    a, b - feature vectors (must be the same length)
  returns the cosine distance, or -1 if lengths mismatch.
*/
float cosineDistance(const std::vector<float> &a, const std::vector<float> &b);

/*
  custom distance for the dumpster CBIR system.
  combines three signals with weighted average:
    - 0.3 * |greenRatio_a - greenRatio_b|
    - 0.3 * histIntersection(colorTexture)
    - 0.4 * cosineDistance(resnet)

  arguments:
    customA, customB - custom feature vectors (greenRatio + colorTexture)
    resnetA, resnetB - ResNet18 embedding vectors for the same images
    colorSize        - size of the color histogram portion (e.g. 512 for bins=8)
  returns the combined distance, or -1 on size mismatch.
*/
float customDumpsterDistance(const std::vector<float> &customA,
                             const std::vector<float> &customB,
                             const std::vector<float> &resnetA,
                             const std::vector<float> &resnetB,
                             int colorSize);

#endif