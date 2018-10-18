#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

// Step 1: complete gradient and threshold
// Step 2: complete sobel
// Step 3: complete canny (recommended substep: return Max instead of C to check it) 

// Raw gradient. No denoising
void gradient(const Mat&Ic, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, CV_BGR2GRAY);

	int m = I.rows, n = I.cols;
	G2 = Mat(m, n, CV_32F);

	Mat Ext;
	copyMakeBorder(I, Ext, 1,1,1,1,BORDER_REPLICATE);

	float diffx, diffy;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			// Compute squared gradient (except on borders)
			// ...
			// G2.at<float>(i, j) = ...
			diffx = float(Ext.at<uchar>(i + 2, j + 1)) - float(Ext.at<uchar>(i, j + 1));
			diffy = float(Ext.at<uchar>(i + 1, j + 2)) - float(Ext.at<uchar>(i + 1, j));
			G2.at<float>(i, j) = static_cast<float>(sqrt(pow(diffx / 2., 2) + pow(diffy / 2., 2)));
		}
	}
}

float Conv2D(const Mat& src, const Mat& kernel, int x, int y){
	float total = 0.0;
	for (int i = 0; i<kernel.rows; i++){
		for (int j = 0; j<kernel.cols; j++){
			total+= float(src.at<uchar>(x+i, y+j)) * kernel.at<int>(i,j);
		}
	}
	return total;
}

// Gradient (and derivatives), Sobel denoising
void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, CV_BGR2GRAY);

	Mat Ext;
	copyMakeBorder(I, Ext, 1,1,1,1,BORDER_REPLICATE);
	//Sobel-X
	int sobelKernelX[3][3] = {
			-1, 0, +1,
			-2, 0, +2,
			-1, 0, +1
	};

	//Sobel-Y
	int sobelKernelY[3][3] = {
			-1, -2, -1,
			0, 0, 0,
			+1, +2, +1
	};

	Mat MatKernelX(3,3, CV_32SC1, sobelKernelX);
	Mat MatKernelY(3,3, CV_32SC1, sobelKernelY);

	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			Ix.at<float>(i, j) = Conv2D(Ext, MatKernelX, i, j)/8;
			Iy.at<float>(i, j) = Conv2D(Ext, MatKernelY, i, j)/8;
			G2.at<float>(i, j) = static_cast<float>(sqrt(pow(Ix.at<float>(i, j), 2) + pow(Iy.at<float>(i, j), 2)));
		}
	}

}

// Gradient thresholding, default = do not denoise
Mat threshold(const Mat& Ic, float s, bool denoise = false)
{
	Mat Ix, Iy, G2;
	if (denoise)
		sobel(Ic, Ix, Iy, G2);
	else
		gradient(Ic, G2);
	int m = Ic.rows, n = Ic.cols;
	Mat C(m, n, CV_8U);
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			if (G2.at<float>(i,j) >= s){
				C.at<uchar>(i, j) = 255;
			}else{
				C.at<uchar>(i, j) = 0;
			}

		}
	}

	return C;
}

// Canny edge detector
Mat canny(const Mat& Ic, float s1)
{
    // s1 small threshold
	Mat Ix, Iy, G2;
	sobel(Ic, Ix, Iy, G2);

	int m = Ic.rows, n = Ic.cols;
	Mat Max(m, n, CV_8U);	// Max pixels ( G2 > s1 && max in the direction of the gradient )
	queue<Point> Q;			// Enqueue seeds ( Max pixels for which G2 > s2 )
    // Initialization queue
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {

			if (G2.at<float>(i,j) >= 3*s1){
                Q.push(Point(j,i)); // Beware: Mats use row,col, but points use x,y
			}
            if (G2.at<float>(i,j) >= s1){
                Max.at<uchar>(i, j) = 255;
            }else{
                Max.at<uchar>(i, j) = 0;
            }

		}
	}
	// Non-max suppression
    float ix,iy;
	double direction;
    for (int i = 1; i < m-1; i++) {
        for (int j = 1; j < n-1; j++) {
            iy = Iy.at<float>(i,j);
            ix = Ix.at<float>(i,j);
            if(ix==0){
                direction = 90.;
            }else{
                direction = atan(iy/ix)*180. /3.1415;
            }

            //Horizontal Edge
            if (((-22.5 < direction) && (direction <= 22.5)) || ((157.5 < direction) && (direction <= -157.5)))
            {
                if ((G2.at<uchar>(i,j) < G2.at<uchar>(i,j+1)) || (G2.at<uchar>(i,j) < G2.at<uchar>(i,j-1)))
                    Max.at<uchar>(i, j) = 0;
            }
            //Vertical Edge
            if (((-112.5 < direction) && (direction <= -67.5)) || ((67.5 < direction) && (direction <= 112.5)))
            {
                if ((G2.at<uchar>(i,j) < G2.at<uchar>(i+1,j)) || (G2.at<uchar>(i,j) < G2.at<uchar>(i-1,j)))
                    Max.at<uchar>(i, j) = 0;
            }

            //-45 Degree Edge
            if (((-67.5 < direction) && (direction <= -22.5)) || ((112.5 < direction) && (direction <= 157.5)))
            {
                if ((G2.at<uchar>(i,j) < G2.at<uchar>(i-1,j+1)) || (G2.at<uchar>(i,j) < G2.at<uchar>(i+1,j-1)))
                    Max.at<uchar>(i, j) = 0;
            }

            //45 Degree Edge
            if (((-157.5 < direction) && (direction <= -112.5)) || ((22.5 < direction) && (direction <= 67.5)))
            {
                if ((G2.at<uchar>(i,j) < G2.at<uchar>(i+1,j+1)) || (G2.at<uchar>(i,j) < G2.at<uchar>(i-1,j-1)))
                    Max.at<uchar>(i, j) = 0;
            }

        }
    }


    // Propagate seeds
	Mat C(m, n, CV_8U);
	C.setTo(0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            if (G2.at<float>(i,j) >= s1*3){
                C.at<uchar>(i, j) = 255;
            }
        }
    }

	while (!Q.empty()) {
		int i = Q.front().y, j = Q.front().x;
        Q.pop();

        if (C.at<float>(i - 1, j - 1) < 1 && Max.at<float>(i - 1, j - 1) > 1)
        {
            C.at<float>(i - 1, j - 1) = 255;
            Q.push(Point(j-1,i-1));
        }
        if (C.at<float>(i + 1, j + 1) < 1 && Max.at<float>(i + 1, j + 1) > 1)
        {
            C.at<float>(i + 1, j + 1) = 255;
            Q.push(Point(j+1,i+1));
        }
        if (C.at<float>(i + 1, j - 1) < 1 && Max.at<float>(i + 1, j - 1) > 1)
        {
            C.at<float>(i + 1, j - 1) = 255;
            Q.push(Point(j-1,i+1));
        }
        if (C.at<float>(i - 1, j + 1) < 1 && Max.at<float>(i - 1, j + 1) > 1)
        {
            C.at<float>(i - 1, j + 1) = 255;
            Q.push(Point(j+1,i-1));
        }
        if (C.at<float>(i, j - 1) < 1 && Max.at<float>(i, j - 1) > 1)
        {
            C.at<float>(i, j - 1) = 255;
            Q.push(Point(j-1, i));
        }
        if (C.at<float>(i, j + 1) < 1 && Max.at<float>(i, j + 1) > 1)
        {
            C.at<float>(i, j + 1) = 255;
            Q.push(Point(j+1,i));
        }
        if (C.at<float>(i + 1, j) < 1 && Max.at<float>(i + 1, j) > 1)
        {
            C.at<float>(i + 1, j) = 255;
            Q.push(Point(j, i+1));
        }
        if (C.at<float>(i - 1, j) < 1 && Max.at<float>(i - 1, j) > 1)
        {
            C.at<float>(i - 1, j) = 255;
            Q.push(Point(j,i-1));
        }

	}

	return C;
}

int main()
{
	Mat I = imread("../road.jpg");

	imshow("Input", I);
	imshow("Threshold", threshold(I, 15));
	imshow("Threshold + denoising", threshold(I, 15, true));
	imshow("Canny", canny(I, 15));
	// Compare
//	  Mat dst, Ic;
//    cvtColor(I, Ic, CV_BGR2GRAY);
//    Canny( Ic, dst, 5, 15, 3 );
//    imshow("opencv canny", dst);

	waitKey();

	return 0;
}
