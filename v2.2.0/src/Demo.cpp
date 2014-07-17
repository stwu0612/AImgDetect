#include "FILE_MACO.h"
#include "TErrorCode.h"
#include "ImageTypeAJudge.h"

#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include "cvaux.h"

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <malloc.h>
#include <pthread.h>

using namespace std;
using namespace cv;
using namespace ImageTypeAJudge_2_2_0;

#define THREAD_NUM    8
vector<string> g_vecFile;
vector<string> g_vecFileRes;

inline long GetIDFromFilePath(const char *filepath)
{
	long ID = 0;
	int  atom =0;
	string tmpPath = filepath;
	for (int i=tmpPath.rfind('/')+1;i<tmpPath.rfind('.');i++)
	{
		atom = filepath[i] - '0';
		if (atom < 0 || atom >9)
			break;
		ID = ID * 10 + atom;
	}
	return ID;
}

inline void PadEnd(char *szPath)
{
	int iLength = strlen(szPath);
	if (szPath[iLength-1] != '/')
	{
		szPath[iLength] = '/';
		szPath[iLength+1] = 0;
	}
}

int imageTypeAJudge(char *szFileList, char *szKeyFile, char *SaveAJudgeRes)
{
	int ret = 0;
	long ImageID = 0;
	long labelID = 0;
	char szImgPath[256];
	FILE *fpListFile = fopen(szFileList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open" << szFileList << endl;
		return 1;
	}
	
	printf("Load Model...\n");
	if(InitImageTypeAJudgeClassify(szKeyFile)){printf("Init ImageTypeAJudge Classify Error!!");return -1;}
	printf("Load Model End!\n");

	FILE* saveFile = fopen(SaveAJudgeRes,"wt");
	if (!saveFile)
	{
		cout << "Can't write file " << endl;
		return 1;
	}
	
	while( EOF != fscanf(fpListFile, "%s %d", szImgPath,&labelID))
	{
		ImageID = GetIDFromFilePath(szImgPath);		
		IplImage* pSrcImg = cvLoadImage(szImgPath,CV_LOAD_IMAGE_ANYDEPTH || CV_LOAD_IMAGE_ANYCOLOR);
		if(!pSrcImg || (pSrcImg->width<16) || (pSrcImg->height<16) || pSrcImg->nChannels != 3 || pSrcImg->depth != IPL_DEPTH_8U) 
		{	
			cvReleaseImage(&pSrcImg);pSrcImg = 0;
			continue;
		}
		
		//宽4字节对齐
		int resize = 310;
		float ratio = 1.0f;
		if (pSrcImg->width > pSrcImg->height) {
			if (pSrcImg->width > resize)
				ratio = resize*1.0 / pSrcImg->width;
		} else if (pSrcImg->height > resize)
			ratio = resize*1.0 / pSrcImg->height;

		int width =  (int)pSrcImg->width * ratio;
		width = (((width + 3) >> 2) << 2); //宽4字节对齐
		int height = (int)pSrcImg->height * ratio;

		IplImage *ipl_resized = cvCreateImage(cvSize(width, height), pSrcImg->depth, pSrcImg->nChannels);
		cvResize(pSrcImg, ipl_resized);
		
		int result = 0;
		int bin = ImageTypeAJudge(ipl_resized,result);

		if (result == 1)//分类结果
		{
			fprintf(saveFile,"%s\n",szImgPath);
		}
		if(pSrcImg){cvReleaseImage(&pSrcImg);pSrcImg = NULL;}
		if(ipl_resized){cvReleaseImage(&ipl_resized);ipl_resized = NULL;}
	}
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}
	if (saveFile) {fclose(saveFile);saveFile = 0;}
	if (ReleaseImageTypeAJudgeClassify()){printf("Release ImageTypeAJudge Classify Error!!");	return -1;}

	return 0;
}

static pthread_mutex_t s_mutex = PTHREAD_MUTEX_INITIALIZER;
typedef struct tagThreadParam
{
	int para;
	int nThreads;
}ThreadParam;

void* ThreadFunc_ImageTypeAJudge(void* para)
{
	int  ret = 0;
	long ImageID;
	int idx;
	int DataLen, vid =0;	
	ThreadParam *pParam = (ThreadParam*)para;

	for (idx = pParam->para; idx < g_vecFile.size(); idx += pParam->nThreads) 
	{
		ImageID = GetIDFromFilePath(g_vecFile[idx].c_str());
		IplImage *pSrcImg = cvLoadImage(g_vecFile[idx].c_str());					//待提取特征图像文件
		if(!pSrcImg || (pSrcImg->width<16) || (pSrcImg->height<16) || pSrcImg->nChannels != 3 || pSrcImg->depth != IPL_DEPTH_8U) 
		{	
			continue;
		}	
		
		//宽4字节对齐
		int resize = 310;
		float ratio = 1.0f;
		if (pSrcImg->width > pSrcImg->height) {
			if (pSrcImg->width > resize)
				ratio = resize*1.0 / pSrcImg->width;
		} else if (pSrcImg->height > resize)
			ratio = resize*1.0 / pSrcImg->height;

		int width =  (int)pSrcImg->width * ratio;
		width = (((width + 3) >> 2) << 2); //宽4字节对齐
		int height = (int)pSrcImg->height * ratio;

		IplImage *ipl_resized = cvCreateImage(cvSize(width, height), pSrcImg->depth, pSrcImg->nChannels);
		cvResize(pSrcImg, ipl_resized);
		
		int result = 0;
		int bin = ImageTypeAJudge(ipl_resized,result);

		if (result == 1)//分类结果
		{
			pthread_mutex_lock(&s_mutex);//对公用文件进行操作，需加锁
			g_vecFileRes.push_back(g_vecFile[idx].c_str());
			pthread_mutex_unlock(&s_mutex);//解锁
		}
		if(pSrcImg){cvReleaseImage(&pSrcImg);pSrcImg = NULL;}
		if(ipl_resized){cvReleaseImage(&ipl_resized);ipl_resized = NULL;}
	}

	printf("thread %d Over!\n", pParam->para);
	pthread_exit(0);
}

int imageTypeAJudge_Multi(char *szFileList, char *szKeyFile, char *SaveAJudgeRes, int nThreads)
{
	int ret = 0;
	long ImageID = 0;
	long labelID = 0;
	char szImgPath[256];

	g_vecFile.clear();
	g_vecFileRes.clear();
	FILE *fpListFile = fopen(szFileList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open" << szFileList << endl;
		return -1;
	}
	while( EOF != fscanf(fpListFile, "%s %d", szImgPath,&labelID))
		g_vecFile.push_back(szImgPath);
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}
	
	printf("Load Model...\n");
	if(InitImageTypeAJudgeClassify(szKeyFile)){printf("Init ImageTypeAJudge Classify Error!!");return -1;}
	printf("Load Model End!\n");
	
	{  //multi-threads part
		pthread_t *pThread = new pthread_t[nThreads];
		ThreadParam *pParam = new ThreadParam[nThreads];
		
		for(int i=0; i<nThreads; ++i)
		{
			pParam[i].para = i;
			pParam[i].nThreads = nThreads;

			pthread_create(pThread+i, NULL, ThreadFunc_ImageTypeAJudge,(void*)(pParam+i));
		}

		for(int i=0; i<nThreads; ++i)
		{
			pthread_join(pThread[i], NULL);
		}
		sleep (1);
		delete [] pThread;
		delete [] pParam;
	}
	
	//写文件
	FILE *fpOut = 0;
	fpOut = fopen(SaveAJudgeRes, "wt+");
	if (!fpOut){cout << "Can't open result file " << SaveAJudgeRes << endl;}

	vector<string>::iterator iterFileRes;
	for(iterFileRes = g_vecFileRes.begin(); iterFileRes != g_vecFileRes.end(); iterFileRes++)
	{		
		fprintf(fpOut,"%s\n",iterFileRes->c_str());
	}
	if (fpOut){fclose(fpOut);fpOut = 0;}
	
	if (ReleaseImageTypeAJudgeClassify()){printf("Release ImageTypeAJudge Classify Error!!");	return -1;}

	return 0;
}

int main(int argc, char* argv[])
{
	int  ret = 0;
	if (argc == 5 && strcmp(argv[1],"-a") == 0)
	{
		ret = imageTypeAJudge(argv[2], argv[3], argv[4]);
	}
	else if (argc == 6 && strcmp(argv[1],"-amuti") == 0)
	{
		ret = imageTypeAJudge_Multi(argv[2], argv[3], argv[4], atol(argv[5]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\t**************ImageTypeAJudge**********************\n" << endl;
		cout << "\tDemo -a ImageList.txt szKeyFile SaveAJudgeRes\n" << endl;
		cout << "\tDemo -amuti ImageList.txt szKeyFile SaveAJudgeRes nThreads\n" << endl;
		return ret;
	}
	return ret;
}

