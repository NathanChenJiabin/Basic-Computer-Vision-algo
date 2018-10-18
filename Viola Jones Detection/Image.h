//
// Created by jiabin on 2018/5/25.
//
#ifndef VIOLADETECTION_IMAGE_H
#define VIOLADETECTION_IMAGE_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

class Image{
public:
    Image(int i, string kind, string name_of_file);

    virtual ~Image();

    const Mat &getPixels() const;

    int getId() const;

    int getLabel() const;

    const string &getFilename() const;

    int **getIntegral_image() const;

    void setIntegral_image();

    int sumPixels(int x, int y, int width, int height);

    int feature1(int x, int y, int width, int height);

    int feature2(int x, int y, int width, int height);

    int feature3(int x, int y, int width, int height);

    int feature4(int x, int y, int width, int height);

    vector<int> calculFeatures(int rank, int nproc);




private:
    Mat pixels;
    int** integral_image;
    int id;
    int label;
    string filename;
};

#endif //VIOLADETECTION_IMAGE_H
