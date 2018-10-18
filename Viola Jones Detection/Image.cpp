//
// Created by jiabin on 2018/5/25.
//

#include "Image.h"

using namespace std;
using namespace cv;


Image::Image(int i, string kind, string name_of_file): filename("/"+name_of_file+"/"+kind+"/im"+to_string(i)+".jpg"),id(i) {
    if(kind.compare("pos")){
        this->label = 1;
    }else{
        this->label = -1;
    }
    this->pixels = imread(filename,IMREAD_GRAYSCALE);
    this->integral_image = new int*[92];

    for(int i=0; i<92; ++i){
        this->integral_image[i] = new int[112];
    }

}

Image::~Image() {

}

const Mat &Image::getPixels() const {
    return pixels;
}

int Image::getId() const {
    return id;
}

int Image::getLabel() const {
    return label;
}

const string &Image::getFilename() const {
    return filename;
}

int **Image::getIntegral_image() const {
    return integral_image;
}

void Image::setIntegral_image(){

    for (int j = 0; j <92 ; ++j) {
        this->integral_image[j][0]=this->pixels.at<uchar>(j,0);

        for (int i = 1; i <112 ; ++i) {
           this->integral_image[j][i] = this->integral_image[j][i-1] + this->pixels.at<uchar>(j,i);
        }
        if(j>=1){
            for (int k = 0; k < 112; ++k) {
                this->integral_image[j][k]+=this->integral_image[j-1][k];
            }
        }

    }

}

int Image::sumPixels(int x, int y, int width, int height){
    return integral_image[x+width-1][y+height-1] + integral_image[x][y] - integral_image[x+width-1][y] - integral_image[x][y+height-1];
}

int Image::feature1(int x, int y, int width, int height){
    return sumPixels(x+width/2, y, width/2, height) - sumPixels(x, y, width/2, height);
}

int Image::feature2(int x, int y, int width, int height){
    return sumPixels(x, y, width, height/2) - sumPixels(x, y+height/2, width, height/2);
}

int Image::feature3(int x, int y, int width, int height){
    int w1 = width/3; int w2 = width-2*w1;
    return sumPixels(x+w1, y, w2, height) - sumPixels(x, y, w1, height) - sumPixels(x+w1+w2, y, w1, height);
}

int Image::feature4(int x, int y, int width, int height){
    return sumPixels(x+width/2, y, width/2, height/2) + sumPixels(x, y+height/2, width/2, height/2)
           - sumPixels(x, y, width/2, height/2) - sumPixels(x+width/2, y+height/2, width/2, height/2);
}

vector<int> Image::calculFeatures(int rank, int nproc){
    vector<int> features;
    for(int width = 6; width<=112; width+=4){
        for(int height = 4; height<=92; height+=4){
            for(int x = 0; x<=112-width; x+=4){
                for(int y = rank*92/nproc; y< (rank+1)*92/nproc - height; y+=4){
                    features.push_back(feature1(x,y,width,height));
                    features.push_back(feature2(x,y,width,height));
                    features.push_back(feature3(x,y,width,height));
                    features.push_back(feature4(x,y,width,height));
                }
            }
        }
    }
    return features;
}



