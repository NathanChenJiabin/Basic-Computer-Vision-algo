#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include "maxflow/graph.h"

#include "image.h"

using namespace std;
using namespace cv;

// This section shows how to use the library to compute a minimum cut on the following graph:
//
//		        SOURCE
//		       /       \
//		     1/         \6
//		     /      4    \
//		   node0 -----> node1
//		     |   <-----   |
//		     |      3     |
//		     \            /
//		     5\          /1
//		       \        /
//		          SINK
//
///////////////////////////////////////////////////

void testGCuts()
{
	Graph<int,int,int> g(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1); 
	g.add_node(2); 
	g.add_tweights( 0,   /* capacities */  1, 5 );
	g.add_tweights( 1,   /* capacities */  6, 1 );
	g.add_edge( 0, 1,    /* capacities */  4, 3 );
	int flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i=0;i<2;i++)
		if (g.what_segment(i) == Graph<int,int,int>::SOURCE)
			cout << i << " is in the SOURCE set" << endl;
		else
			cout << i << " is in the SINK set" << endl;
}


Image<uchar> segementation(const Image<Vec3b>& img){
    Image<uchar>G;
    cvtColor(img, G, CV_BGR2GRAY);
    Image<float> F;
    G.convertTo(F, CV_32F);
    int w = img.width();
    int h = img.height();
    Graph<double,double,double> g(/*estimated # of nodes*/ w*h, /*estimated # of edges*/ (w-1)*h+(h-1)*w);
    g.add_node(w*h);
    int y, x;
    double coeff, diff;
    double dist_s, dist_t;
    double cte = 500.;
    Vec3b pixel;
    Vec3b source(255,255,255);
    Vec3b sink(0,160,160);
    for (int k=0; k<w*h; k++){
        y = k / w;
        x = k % w;
        if(y<h-1){
            diff = F(Point(x, y)) - F(Point(x, y+1));
            coeff = cte / (1. + pow(diff, 2));
            g.add_edge(k, (y+1)*w+x, coeff, coeff);
        }
        if(x<w-1){
            diff = F(Point(x, y)) - F(Point(x+1, y));
            coeff = cte / (1. + pow(diff, 2));
            g.add_edge(k, y*w+x+1, coeff, coeff);
        }

        pixel = img(Point(x, y));
        dist_s = norm(pixel, source);
        dist_t = norm(pixel, sink);
        g.add_tweights(k, dist_s, dist_t);

    }
    Image<uchar> seg(w, h);
    double flow = g.maxflow();
    cout << "Flow = " << flow << endl;
    for (int k=0;k<w*h;k++){
        if (g.what_segment(k) == Graph<double,double,double>::SOURCE)
            seg(Point(k%w, k/w))=0;
        else
            seg(Point(k%w, k/w))=255;
    }

    return seg;

}

int main() {
	//testGCuts();

	Image<Vec3b> I= Image<Vec3b>(imread("../fishes.jpg"));
	imshow("I",I);
	Image<uchar> result = segementation(I);
	imshow("result", result);
	waitKey(0);
	return 0;
}
