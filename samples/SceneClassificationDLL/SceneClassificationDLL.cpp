// SceneClassificationDLL.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

#include "constant.h"
string root_path;
Mat Ftrain_hog2x2, Ftrain_colorhist, Vocabulary, kernel_test;
struct svm_model* svm_models[NUM_CLASS_USE];
string * class_name;
int * class_num;


extern "C" __declspec(dllexport)
int GetSceneName(int scene_label, char *sScenName)
{
	string pepleclass = "People";
	if (scene_label >= 0)
		memcpy(sScenName, class_name[scene_label].c_str(), class_name[scene_label].length());
	else if (scene_label == -2)
		memcpy(sScenName, pepleclass.c_str(), pepleclass.length());
	else return -1;
	return 0;
}
