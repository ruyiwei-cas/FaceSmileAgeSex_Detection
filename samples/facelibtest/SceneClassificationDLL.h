
#include <opencv\cv.h>
#include <opencv2\core\core.hpp>
using namespace cv;

extern "C" __declspec(dllexport)
void initialize_scene(char *path);


extern "C" __declspec(dllexport)
double test_single_img(Mat im);

extern "C" __declspec(dllexport)
int GetSceneName(int scene_label, char *sScenName);

extern "C" __declspec(dllexport)
void destroy_scene();