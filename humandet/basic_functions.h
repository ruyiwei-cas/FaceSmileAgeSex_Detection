#pragma once

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <ctype.h>
#include <io.h>

#ifdef __linux__
#include <unistd.h>
#else
#include <direct.h>
#endif

#include  "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace std;
using namespace cv;


void readDirectory(string dirName, vector<string> &fileList, bool isTopDir=true, bool useRelativePath=false);
bool splitPath(string path, string &dir, string &name);
bool checkDir(string dir);
bool saveImg(string filename, const Mat &im);
void myResize(Mat &im, int maxSize);


