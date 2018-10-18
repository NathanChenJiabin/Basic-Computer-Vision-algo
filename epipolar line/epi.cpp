#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include "image.h"

struct Camera {
	Matx33d A;
	Vec3d b;
	void read(string name) {
		ifstream f;
		f.open(name);
		if (!f.is_open()) {
			cout << "Cannot read camera file" << endl;
			return;
		}
		for (int i = 0; i < 3; i++)
			f >> A(i, 0) >> A(i, 1) >> A(i, 2) >> b[i];
		f.close();
	}
	void print() const {
		cout << "A= " << A << endl
			<< "b= " << b << endl;
	}
	Vec3d center() const {
		return Vec3d((-A.inv()*b).val);
	}
	Vec3d proj(const Vec3d& M) const {
		return Vec3d((A*M + b).val);
	}
};

Matx33d fundamental(const Camera& C1, const Camera& C2) {
	Vec3d e2 = C2.proj(C1.center());
	Matx33d E2(0, -e2[2], e2[1],
		e2[2], 0, -e2[0],
		-e2[1], e2[0], 0);
	return E2 * C2.A*C1.A.inv();
}

struct Data {
	Image<Vec3b> I1, I2;
	Image<float> F1, F2;
	Camera C1, C2;
	Matx33d F;
};

void onMouse1(int event, int x, int y, int foo, void* p)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	Point m1(x, y);

	Data* D = (Data*)p;
	circle(D->I1, m1, 2, Scalar(0, 255, 0), 2);
	imshow("I1", D->I1);

	Vec3d m1p(m1.x, m1.y, 1);
	// Epipolar line equation 
	Vec3d l = D->F*m1p;

	// Todo:

	// 1 - compute two points on the epipolar line and draw it
	//			Point m2a(0,????),m2b(D->I2.width(),????);
	//			line(D->I2,m2a,m2b,Scalar(0,255,0),1);
    Point m2a(0, int(-l[2]/l[1]));
    int w = D->I2.width();
    Point m2b(w, int((-l[2] - w*l[0])/l[1]));
    line(D->I2, m2a, m2b, Scalar(0,255,0), 1);

	// 2 - find the point on the epiplar line that best correlates with the clicked point and draw it
	//			Use double NCC(const Image<float>& I1,Point m1,const Image<float>& I2,Point m2,int n);
	//			circle(D->I2,???,2,Scalar(0,255,0),2);
    double meilleurNCC = -2.0;
    double ncc;
    Point matchPoint;
    for (int k = 0; k < w; k++){
        Point tmp(k, int((-l[2] - k*l[0])/l[1]) );
        ncc = NCC(D->F1, m1, D->F2, tmp, 8);
        if (ncc > meilleurNCC){
            meilleurNCC = ncc;
            matchPoint = tmp;
        }

    }
    circle(D->I2, matchPoint, 2, Scalar(0,255,0),2);

	imshow("I2", D->I2);
}

void onMouse2(int event, int x, int y, int foo, void* p)
{
	// 3 - From image 2 to image 1
	//	   Do not forget that F can also be used from 2 to 1. No need to compute a new one!
    if (event != CV_EVENT_LBUTTONDOWN)
        return;
    Point m1(x, y);

    Data* D = (Data*)p;
    circle(D->I2, m1, 2, Scalar(0, 255, 0), 2);
    imshow("I2", D->I2);

    Vec3d m1p(m1.x, m1.y, 1);
    // Epipolar line equation
    Vec3d l = D->F.t()*m1p;

    Point m2a(0, int(-l[2]/l[1]));
    int w = D->I1.width();
    Point m2b(w, int((-l[2] - w*l[0])/l[1]));
    line(D->I1, m2a, m2b, Scalar(0,255,0), 1);

    double meilleurNCC = -2.0;
    double ncc;
    Point matchPoint;
    for (int k = 0; k < w; k++){
        Point tmp(k, int((-l[2] - k*l[0])/l[1]) );
        ncc = NCC(D->F2, m1, D->F1, tmp, 8);
        if (ncc > meilleurNCC){
            meilleurNCC = ncc;
            matchPoint = tmp;
        }

    }
    circle(D->I1, matchPoint, 2, Scalar(0,255,0),2);

    imshow("I1", D->I1);
}

int main(int argc, char** argv)
{
	Data D;
	D.I1 = imread("../face00.tif");
	D.I2 = imread("../face01.tif");
	imshow("I1", D.I1);
	imshow("I2", D.I2);

	D.C1.read("../face00.txt");
	D.C2.read("../face01.txt");
	D.C1.print();
	D.C2.print();

	D.F = fundamental(D.C1, D.C2);
	cout << "F= " << D.F << endl;

	Image<uchar>G1, G2;
	cvtColor(D.I1, G1, CV_BGR2GRAY);
	cvtColor(D.I2, G2, CV_BGR2GRAY);
	G1.convertTo(D.F1, CV_32F);
	G2.convertTo(D.F2, CV_32F);

	setMouseCallback("I1", onMouse1, &D);
	setMouseCallback("I2", onMouse2, &D);

	waitKey(0);
	return 0;
}
