/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 2: Content-Based Image Retrieval

  distance metric implementations.
*/

#include "distances.h"
#include <algorithm>
#include <cmath>

float ssd(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size())
        return -1.0f;

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float histIntersection(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size())
        return -1.0f;

    // compute the sum of each histogram (needed to normalize)
    float sumA = 0.0f, sumB = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
    {
        sumA += a[i];
        sumB += b[i];
    }
    if (sumA == 0.0f || sumB == 0.0f)
        return 1.0f; // edge case: empty histogram

    // compute intersection of the normalized histograms
    // intersection = sum_i min(a_i / sumA, b_i / sumB)
    float intersection = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
    {
        float aNorm = a[i] / sumA;
        float bNorm = b[i] / sumB;
        intersection += std::min(aNorm, bNorm);
    }

    // convert similarity to distance (so smaller = more similar,
    // consistent with the convention SSD uses)
    return 1.0f - intersection;
}

float histIntersectionMulti(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size() || a.size() % 2 != 0)
        return -1.0f;

    // split each vector into two halves
    size_t half = a.size() / 2;
    std::vector<float> aTop(a.begin(), a.begin() + half);
    std::vector<float> aBot(a.begin() + half, a.end());
    std::vector<float> bTop(b.begin(), b.begin() + half);
    std::vector<float> bBot(b.begin() + half, b.end());

    // run regular histogram intersection on each half, then average
    float dTop = histIntersection(aTop, bTop);
    float dBot = histIntersection(aBot, bBot);

    if (dTop < 0 || dBot < 0)
        return -1.0f;

    // equal weighting
    return 0.5f * dTop + 0.5f * dBot;
}

float colorTextureDistance(const std::vector<float> &a, const std::vector<float> &b,
                           int colorSize)
{
    if (a.size() != b.size() || (int)a.size() < colorSize)
        return -1.0f;

    // split each feature into color part and texture part
    std::vector<float> aColor(a.begin(), a.begin() + colorSize);
    std::vector<float> aTex(a.begin() + colorSize, a.end());
    std::vector<float> bColor(b.begin(), b.begin() + colorSize);
    std::vector<float> bTex(b.begin() + colorSize, b.end());

    float dColor = histIntersection(aColor, bColor);
    float dTex = histIntersection(aTex, bTex);
    if (dColor < 0 || dTex < 0)
        return -1.0f;

    // equal weighting
    return 0.5f * dColor + 0.5f * dTex;
}

float cosineDistance(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size())
        return -1.0f;

    // compute dot product and the L2 norms in one pass
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
    {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if (normA == 0.0f || normB == 0.0f)
        return 1.0f; // edge case: zero vector

    // cosine similarity = dot / (||a|| * ||b||)
    float cosSim = dot / (std::sqrt(normA) * std::sqrt(normB));

    // convert to distance (smaller = more similar, consistent with our convention)
    return 1.0f - cosSim;
}

float customDumpsterDistance(const std::vector<float> &customA,
                             const std::vector<float> &customB,
                             const std::vector<float> &resnetA,
                             const std::vector<float> &resnetB,
                             int colorSize)
{
    if (customA.size() != customB.size())
        return -1.0f;
    if (resnetA.size() != resnetB.size())
        return -1.0f;
    if (customA.size() < 2)
        return -1.0f;

    // green ratio difference (first element of each custom vector)
    float greenDiff = std::abs(customA[0] - customB[0]);

    // color+texture distance (the rest of the custom vector is [color (colorSize), texture])
    // pull the [1..end] portion out for histIntersection-style split
    std::vector<float> ctA(customA.begin() + 1, customA.end());
    std::vector<float> ctB(customB.begin() + 1, customB.end());
    float ctDist = colorTextureDistance(ctA, ctB, colorSize);
    if (ctDist < 0)
        return -1.0f;

    // resNet cosine distance
    float resnetDist = cosineDistance(resnetA, resnetB);
    if (resnetDist < 0)
        return -1.0f;

    // weighted combination
    return 0.3f * greenDiff + 0.3f * ctDist + 0.4f * resnetDist;
}