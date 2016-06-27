#include "../Platform.h"

#include "basic_functions.h"


/* Read the file list in a folder
 *
 * dirName: the input directory name
 * fileList: the list names of files in the directory 
 * isSubDir: whether the 'dirName' is a top-directory, (i.e., not a sub-directory)
 * relativePath: whether use relative path w.r.t dirName
 */
void readDirectory(string dirName, vector<string> &fileList, bool isTopDir, bool useRelativePath)
{
#ifdef __win__
	if(isTopDir)	// clear 'fileList' if 'dirName' is NOT a top-directory
		fileList.clear();

	_finddata_t findData;
	intptr_t findHandle;
	if( (findHandle=_findfirst((dirName+"/*.*").c_str(),&findData)) != -1L)
	{
		string fileName;
		do
		{
			fileName=findData.name;
			if (fileName=="." || fileName=="..")	// skip current/parent directory
				continue;

			fileName=dirName + "/" + fileName;	// add the directory to the file path
			if (findData.attrib & _A_SUBDIR)	// whether it's a sub directory
				readDirectory(fileName, fileList, false);
			else
			{
				string fileExtension=fileName.substr(fileName.length()-3);
				if (fileExtension=="jpg" || fileExtension=="JPG")
					fileList.push_back(fileName);
			}
		}while(_findnext(findHandle, &findData) == 0L);
	}
	_findclose(findHandle);

	if(useRelativePath)
	{
		int lengTopDir = dirName.size() + 1;
		for(vector<string>::iterator iter=fileList.begin(); iter!=fileList.end(); ++iter)
			iter->erase(0, lengTopDir);
	}
#endif
	return;
}


/* split the path to directory and name
 */
bool splitPath(string path, string &dir, string &name)
{
	int idx = path.rfind('/');
	dir = path.substr(0, idx);
	name = path.substr(idx+1);

	return (idx != path.size());
}


/* check the directory.
 * if the directory doesn't exists, then create it recursively.
 */
bool checkDir(string dir)
{	
#ifdef __win__
	if(_access(dir.c_str(),0) != -1)	// dir already exists
		return true;

	string parent, self;
	if(splitPath(dir,parent,self) && !checkDir(parent))
		return false;
	else
		return (_mkdir(dir.c_str()) == 0);
#endif
#ifdef __linux__
	return true;
#endif
}


/* save the image to the specified path.
 * Note that if the directory of the path doesn't exist, create it.
 */
bool saveImg(string filename, const Mat &im)
{
	string dir, name;
	if(splitPath(filename,dir,name) && !checkDir(dir))
		return false;
	else
		return imwrite(filename, im);
}


/* resize the im such that its width and height not exceeding the maxSize
 */
void myResize(Mat &im, int maxSize)
{
	int width = im.cols;
	int height = im.rows;
	int maxWH = (width>height? width:height);
	if(maxWH > maxSize)
	{
		double ratio = 1.0 * maxSize / maxWH;
		int resizeW = cvRound(width * ratio);
		int resizeH = cvRound(height * ratio);
		resize(im, im, Size(resizeW, resizeH));
	}
	return;
}

