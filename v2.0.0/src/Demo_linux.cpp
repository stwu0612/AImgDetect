#include "ColorLayout.h"
#include "EdgeHist.h"
#include "FILE_MACO.h"
#include "Process.h"
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

using namespace std;
using namespace cv;
using namespace ImageTypeAJudge_2_0_0;

int main()
{
	int num_sample = 10;//A:1500;
//	float *response = NULL;response = (float *)malloc(num_sample*sizeof(float));
	int pos_num=0;
	int neg_num=0;

	FILE* saveFile = NULL;
	saveFile = fopen("./save.txt","w+");
	fprintf(saveFile,"imageID		result\n");

	printf("Load Model...\n");
	char* featPath = "./feat/feat.xml";
	if(InitImageTypeAJudgeClassify(featPath)){printf("Init ImageTypeAJudge Classify Error!!");return -1;}
	printf("Load Model End!\n");

	ifstream readfile("/home/chigo/ImageTypeAJudgev2.0.0/img/test_pos1500/list.txt",ios::in);
	
	for(int i=0;i<num_sample;i++)
	{
		char filename[100] = {0};
		readfile.getline(filename,100);
		cout<<filename<<endl;
		char filename1[1024] = {0};
		strcpy(filename1,"/home/chigo/ImageTypeAJudgev2.0.0/img/test_pos1500/");		
		strcat(filename1,filename);
		
		///////////////////////read and resize the data of the picture//////
					
		IplImage* pSrcImg = cvLoadImage(filename1,CV_LOAD_IMAGE_ANYDEPTH || CV_LOAD_IMAGE_ANYCOLOR);
		double t = (double)getTickCount();
		
		int result = 0;
		if (ImageTypeAJudge(pSrcImg,result)){printf("Image Splicing Error");return TEC_UNSUPPORTED;}
		if (result == 1)//分类结果
		{
			fprintf(saveFile,"%s		1\n",filename);
			pos_num++;	
		}
		else if (result == 0)
		{	
			fprintf(saveFile,"%s		0\n",filename);
			neg_num++;
		}	
		if(pSrcImg){cvReleaseImage(&pSrcImg);pSrcImg = NULL;}	
		t = (double)getTickCount() - t;
		t = t*1000./cv::getTickFrequency();
		printf("detection time = %.2fms\n", t);
	}
	double pos = pos_num*1.0/num_sample;
	double neg = neg_num*1.0/num_sample;
	fprintf(saveFile,"num_sample=%d,pos_num=%d,pos=%.4f,neg_num=%d,neg=%.4f\n",num_sample,pos_num,pos,neg_num,neg);
	printf("num_sample=%d,pos_num=%d,pos=%.4f,neg_num=%d,neg=%.4f\n",num_sample,pos_num,pos,neg_num,neg);
	fclose(saveFile);

//	free(&response);
	if (ReleaseImageTypeAJudgeClassify()){printf("Release ImageTypeAJudge Classify Error!!");	return -1;}

	return 0;
}
