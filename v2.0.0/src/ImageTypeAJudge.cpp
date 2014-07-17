#pragma once
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

#if _WIN32
#include  <io.h>
#else
#include  <unistd.h>
#endif

using namespace std;
using namespace cv;

namespace ImageTypeAJudge_2_0_0
{
	CvSVM imagetypeajudge_svm;

int InitImageTypeAJudgeClassify(char* featPath)
{
	if(featPath == NULL)
	{
		printf("feat file is null!!\n");
		return TEC_INVALID_PARAM;
	}
	if(sizeof(char)*sizeof(featPath) > 1024)
	{
		printf("feat path is too long!!\n");
		return TEC_INVALID_PARAM;
	}

	int bFileEx = -1;
	bFileEx = access(featPath,0);

	if(bFileEx != 0)
	{
		printf("feat file is not existed\n");
		return TEC_INVALID_PARAM;
	}

	imagetypeajudge_svm.clear();

	FILE* pSVMFile = NULL;
	pSVMFile = fopen(featPath,"r");
	if(pSVMFile == NULL)
	{
		printf("svm file is null\n");
		return TEC_INVALID_PARAM;
	}

	char feat_content[1024];
	memset(feat_content,0,sizeof(char)*1024);
	int i = 0;
	while(fscanf(pSVMFile,"%s",feat_content) != EOF)
	{
		if(i == 0)
		{
			if(strcmp(feat_content,"<?xml") != 0)
			{
				fclose(pSVMFile);
				pSVMFile = NULL;
				printf("the format of the svm feat is not right!!\n");
				return TEC_INVALID_PARAM;
			}
		}
		if(i == 2)
		{
			if(strcmp(feat_content,"<opencv_storage>") != 0)
			{	
				fclose(pSVMFile);
				pSVMFile = NULL;
				printf("the format of the svm feat is not right!!\n");
				return TEC_INVALID_PARAM;
			}
		}
		i ++;

		if(i > 2)
			break;
	}
	fclose(pSVMFile);
	pSVMFile = NULL;

	if(i == 0)
	{
		cout <<"Image Splicing Feat is Invalid"<<endl;
		return TEC_INVALID_PARAM;
	}

	imagetypeajudge_svm.load(featPath);

	return TOK;
}

//区域灰度像素差统计
//Add by Chigo
float Region_Detect(IplImage* src)
{
	IplImage* gray = NULL;
	gray = cvCreateImage(cvSize(src->width,src->height),src->depth,1);
	cvCvtColor(src,gray,CV_BGR2GRAY);
	cvSmooth(gray, gray, CV_MEDIAN, 3, 0, 0, 0);//中值滤波
	IplImage* src_region = NULL;
	src_region = cvCreateImage(cvSize(src->width,src->height),src->depth,1);
	int x1 = int(src->width*0.125);
	int x2 = int(src->width*0.875);
	int y1 = int(src->height*0.125);
	int y2 = int(src->height*0.875);
	int value_region = 0;
	int num_region = 0;
	for (int i=0;i<src->height;i++)
	{
		for (int j=0;j<src->width;j++)
		{
			if (i>y1&&i<y2&&j>x1&&j<x2)
			{
				((uchar *)(src_region->imageData + i*src_region->widthStep))[j] = 0;
			}
			else
			{
				int value_gray = ((uchar *)(gray->imageData + i*gray->widthStep))[j];
				((uchar *)(src_region->imageData + i*src_region->widthStep))[j] = value_gray;
				value_region += value_gray;
				num_region++;
			}
		}
	}
	//获取区域灰度均值
	int mean_region = int(value_region*1.0/num_region);
	int num_reduce = 0;
	for (int i=0;i<src->height;i++)
	{
		for (int j=0;j<src->width;j++)
		{			
			if (i<y1||i>y2||j<x1||j>x2)
			{
				int value_gray = ((uchar *)(gray->imageData + i*gray->widthStep))[j];
				int value_reduce = abs(value_gray - mean_region);
				if (value_reduce>50)
				{
					num_reduce++;
				}
			}
		}
	}
	float rai_reduce = num_reduce*1.0/num_region;

	if(gray){cvReleaseImage(&gray);gray = NULL;}
	if(src_region){cvReleaseImage(&src_region);src_region = NULL;}	

	return rai_reduce;
}

//blur 
//reference No-reference Image Quality Assessment using blur and noisy
//write by Min Goo Choi, Jung Hoon Jung  and so on  
int ImageTypeAJudge_Blur(IplImage* pSrcImg,float fBlur[])
{
	memset(fBlur,0,sizeof(float)*BLURDIM);

	if(!pSrcImg) return -1;

	IplImage* pGrayImg = NULL;
	pGrayImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	//for mean filter
	IplImage* pNoisyImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);

	if(pSrcImg->nChannels == 3) cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);
	else cvCopy(pSrcImg,pGrayImg);

	//something different form paper i use opencv median filter here
	cvSmooth(pGrayImg,pNoisyImg,CV_MEDIAN);

	int nHei = pGrayImg->height; int nWid = pGrayImg->width;

	int total = (nWid)*(nHei);

	int iLineBytes = pGrayImg->widthStep;
	uchar* pData = (uchar*)pGrayImg->imageData;

	int iNoisyBytes = pNoisyImg->widthStep;
	uchar* pNoisyData = (uchar*)pNoisyImg->imageData;

	int steps = 0;
	//result
	//blur
	double blur_mean = 0;
	double blur_ratio = 0;
	//noisy
	double nosiy_mean = 0;
	double nosiy_ratio = 0;

	//means DhMean and DvMean in paper
	//for edge
	// it is global mean in paper i will try local later
	double ghMean = 0; 
	double gvMean = 0;
	//for noisy
	double gNoisyhMean = 0;
	double gNoisyvMean = 0;
	//Nccand-mean
	double gNoisyMean = 0;

	//tmp color value for h v
	double ch = 0;	
	double cv = 0;
	//The Thresh blur value best detected
	const double blur_th = 0.1;
	//blur value sum
	double blurvalue = 0;
	//blur count
	int blur_cnt = 0;
	//edge count
	int h_edge_cnt = 0;
	int v_edge_cnt = 0;
	//noisy count
	int noisy_cnt = 0;
	// noisy value
	double noisy_value = 0;

	//mean Dh(x,y) in the paper 
	// in code it means Dh(x,y) and Ax(x,y)
	double* phEdgeMatric = new double[total];
	double* pvEdgeMatric = new double[total];
	// for noisy
	//Dh Dv in the paper
	double* phNoisyMatric = new double[total];
	double* pvNoisyMatric = new double[total];
	//Ncond in the paper
	double * NoisyM = new double[total];

	//means Ch(x,y) Cv(x,y) in the paper
	double* tmpH = new double[total];
	double* tmpV = new double[total];


	//for blur and noisy
	//loop 1
	for(int i = 0; i < nHei; i ++)
	{
		uchar* pOffset = pData;
		uchar* pNoisyOff = pNoisyData;
		steps = i*nWid;	

		for(int j = 0; j < nWid; j ++)
		{	
			int nSteps = steps + j;
			if(i == 0 || i == nHei -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
				//for noisy
				phNoisyMatric[nSteps] = 0;
				pvNoisyMatric[nSteps] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
				//for edge
				phEdgeMatric[nSteps] = 0;
				pvEdgeMatric[nSteps] = 0;
				//for noisy
				phNoisyMatric[nSteps] = 0;
				pvNoisyMatric[nSteps] = 0;
			}
			else
			{
				//for edge
				ch = abs(*(pOffset-1) - *(pOffset+1)) * 1.0 / 255.0;
				phEdgeMatric[nSteps] = ch;
				ghMean += ch;

				cv = abs(*(pOffset-nWid) - *(pOffset+nWid)) * 1.0 / 255.0;
				pvEdgeMatric[nSteps] = cv;
				gvMean += cv;

				//for noisy
				ch = abs(*(pNoisyOff-1) - *(pNoisyOff+1)) * 1.0 / 255.0;
				phNoisyMatric[nSteps] = ch;
				gNoisyhMean += ch;
				cv = abs(*(pNoisyOff-nWid) - *(pNoisyOff+nWid)) * 1.0 / 255.0;
				pvNoisyMatric[nSteps] = cv;
				gNoisyvMean += cv;
			}

			double tmp_blur_value = 0;
			double tmp_ch = 0;
			double tmp_cv = 0;
			ch = (phEdgeMatric[nSteps] / 2);
			if(ch != 0)
				tmp_ch = abs((*pOffset) * 1.0 / 255 - ch) * 1.0 / ch;	
			cv = (pvEdgeMatric[nSteps] / 2);
			if(cv != 0)
				tmp_cv = abs((*pOffset) * 1.0 / 255 - cv) * 1.0 / cv;

			tmp_blur_value = max(tmp_ch,tmp_cv);
			//	blurvalue += tmp_blur_value;
			if(tmp_blur_value > blur_th) 
			{
				blur_cnt ++;
				blurvalue += tmp_blur_value;
			}

			pOffset ++;
			pNoisyOff ++;
		}
		pData += iLineBytes;
		pNoisyData += iNoisyBytes;
	}

	//for edge and noisy
	//for edge
	ghMean /= (total);
	gvMean /= (total);	
	//noisy
	gNoisyhMean /= total;
	gNoisyvMean /= total;

	//loop 2
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*nWid;
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			ch = phEdgeMatric[nSteps];
			tmpH[nSteps] = ch > ghMean ?  ch : 0;
			cv = pvEdgeMatric[nSteps];
			tmpV[nSteps] = cv > gvMean ?  cv : 0;

			ch = phNoisyMatric[nSteps];
			cv = pvNoisyMatric[nSteps];
			if(ch <= gNoisyhMean && cv <= gNoisyvMean)
			{
				NoisyM[nSteps] = max(ch,cv);
			}
			else
				NoisyM[nSteps] = 0;

			gNoisyMean += NoisyM[nSteps];
		}
	}
	gNoisyMean /= total;

	//loop 3
	for(int i = 0; i < nHei; i ++)
	{
		steps = i*(nWid);
		for(int j = 0; j < nWid; j ++)
		{
			int nSteps = steps + j;
			//for edge
			if(i == 0 || i == nHei -1)
			{
				//	phEdge[steps+j] = 0;
				//	pvEdge[steps+j] = 0;
			}
			else if(j == 0 || j == nWid -1)
			{
				//	phEdge[steps+j] = 0;
				//	pvEdge[steps+j] = 0;
			}
			else
			{
				//for edge
				if(tmpH[nSteps] > tmpH[nSteps-1] && tmpH[nSteps] > tmpH[nSteps+1])
				{
					//	phEdge[steps+j] = 1;
					h_edge_cnt ++;
				}
				//else phEdge[steps+j] = 0;

				if(tmpV[nSteps] > tmpV[steps-nWid] && tmpV[nSteps] > tmpV[steps+nWid])
				{
					//	pvEdge[steps+j] = 1;
					v_edge_cnt ++;
				}
				//	else pvEdge[steps+j] = 0;

				if(NoisyM[nSteps] > gNoisyMean)
				{
					noisy_cnt++;
					noisy_value += NoisyM[nSteps];
				}

			}

		}
	}

	blur_mean = blurvalue * 1.0 / blur_cnt;
	blur_ratio = blur_cnt * 1.0 / (h_edge_cnt+v_edge_cnt);

	nosiy_mean = noisy_value * 1.0 / noisy_cnt;
	nosiy_ratio = noisy_cnt * 1.0 / total;

	//the para is provided by paper
	//another para 1.55 0.86 0.24 0.66
	double gReulst = 1 -(blur_mean + 0.95 * blur_ratio + \
		nosiy_mean * 0.3 + 0.75 * nosiy_ratio);

	fBlur[0] = blur_mean;
	fBlur[1] = blur_ratio;
	fBlur[2] = nosiy_mean;
	fBlur[3] = nosiy_ratio;
	fBlur[4] = gReulst;

	if(pGrayImg){cvReleaseImage(&pGrayImg);pGrayImg = NULL;}
	if(pNoisyImg){cvReleaseImage(&pNoisyImg);pNoisyImg = NULL;}
	if(phEdgeMatric){delete []phEdgeMatric; phEdgeMatric = NULL;}
	if(pvEdgeMatric){delete []pvEdgeMatric; pvEdgeMatric = NULL;}
	if(phNoisyMatric){delete []phNoisyMatric; phNoisyMatric = NULL;}
	if(pvNoisyMatric){delete []pvNoisyMatric; pvNoisyMatric = NULL;}
	if(NoisyM){delete []NoisyM; NoisyM = NULL;}
	if(tmpH){delete []tmpH; tmpH = NULL;}
	if(tmpV){delete []tmpV; tmpV = NULL;}
	return TOK;
}

//ImageTypeAJudge
//Add by Chigo
int ImageTypeAJudge(IplImage* pSrcImg,int &result)
{
	if(NULL == pSrcImg) {printf("Please input image!!");return TEC_INVALID_PARAM;}
	if(3 != pSrcImg->nChannels) {printf("Please input 3 channels image!!");return TEC_INVALID_PARAM;}
	result = 0;//初始化检测结果
	Feat feat;	
	int image_size = 128;//归一化后图像尺寸
	IplImage* img_resize_1 = NULL;
	IplImage* img_roi = NULL;
	int nWid = pSrcImg->width;
	int nHei = pSrcImg->height;
	int maxWid = max(nWid,nHei);
	int minHei = min(nWid,nHei);
	if(16>minHei){printf("image size is too small !!!\n");return TEC_INVALID_PARAM;}
	if(10000<maxWid){printf("image size is too large !!!\n");return TEC_INVALID_PARAM;}
	if(minHei*10<maxWid){printf("image size is not normal !!!\n");return TEC_INVALID_PARAM;}
	if (minHei==nWid)
	{				
		nHei = nHei*image_size/nWid; nWid = image_size;		
		img_resize_1 = cvCreateImage(cvSize(nWid,nHei),pSrcImg->depth,pSrcImg->nChannels);
		cvResize(pSrcImg,img_resize_1,CV_INTER_CUBIC);				
		img_roi = cvCloneImage(img_resize_1);
		cvSetImageROI(img_roi,cvRect(0, int((nHei-nWid)/2), nWid, nWid));
	}
	else
	{
		nWid = nWid*image_size/nHei; nHei = image_size;			
		img_resize_1 = cvCreateImage(cvSize(nWid,nHei),pSrcImg->depth,pSrcImg->nChannels);
		cvResize(pSrcImg,img_resize_1,CV_INTER_CUBIC);
		img_roi = cvCloneImage(img_resize_1);
		cvSetImageROI(img_roi,cvRect(int((nWid-nHei)/2),0, nHei,nHei));
	}			
	IplImage* src_roi= cvCreateImage(cvSize(image_size,image_size),pSrcImg->depth,pSrcImg->nChannels);
	cvCopy( img_roi, src_roi, NULL );			

	//区域检测进行初选		
	float value_region = Region_Detect(img_resize_1);
	if(value_region>0.4)//区域统计
	{
		//图片为非A类图
		result = 0;
	}
	else
	{				
		CvMat *sample_all = cvCreateMat(1,sizeof(Feat)/sizeof(float),CV_32FC1);
		cvSetZero(sample_all);

		//Blur
		ImageTypeAJudge_Blur(src_roi,feat.fBlur);
		//CLD
		ColorLayoutExtractor(src_roi,feat.fCLD);

		IplImage* pSobelImg =  NULL; IplImage* pGrayImg = NULL;
		pSobelImg = cvCreateImage(cvGetSize(src_roi),src_roi->depth,1);
		pGrayImg = cvCreateImage(cvGetSize(src_roi),src_roi->depth,1);
		//get sobel image
		if(src_roi->nChannels == 3){cvCvtColor(src_roi,pGrayImg,CV_BGR2GRAY);}
		else{cvCopy(src_roi,pGrayImg);}	
		getSobel(pGrayImg,pSobelImg);	

		//EHD
		IplImage* pWaterImg = NULL;
		pWaterImg = cvCreateImage(cvGetSize(src_roi),src_roi->depth,1);
		EdgeHistExtractor(pGrayImg,feat.fEHD);
		EdgeHistExtractor(pSobelImg,feat.fsobelEHD);

		cvDilate(pGrayImg,pWaterImg);
		cvErode(pWaterImg,pWaterImg);
		cvAbsDiff(pGrayImg,pWaterImg,pWaterImg);

		cvThreshold(pWaterImg,pWaterImg,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
		EdgeHistExtractor(pWaterImg,feat.fOptEHD);

		cvThreshold(pSobelImg,pGrayImg,0,255,CV_THRESH_OTSU | CV_THRESH_BINARY);
		EdgeHistExtractor(pGrayImg,feat.fBinEHD);

		//特征向量转换
		for (int j=0;j<sample_all->cols;j++)
		{
			if (j<EHDDIM)
			{
				CV_MAT_ELEM(*sample_all, float, 0, j ) = feat.fEHD[j];
			}
			else if(j<(EHDDIM*2))
			{
				CV_MAT_ELEM(*sample_all, float, 0, j ) = feat.fsobelEHD[j-EHDDIM];
			}
			else if(j<(EHDDIM*3))
			{
				CV_MAT_ELEM(*sample_all, float, 0, j ) = feat.fBinEHD[j-EHDDIM*2];
			}
			else if(j<(EHDDIM*4))
			{
				CV_MAT_ELEM(*sample_all, float, 0, j ) = feat.fOptEHD[j-EHDDIM*3];
			}
			else if(j<(EHDDIM*4+CLDDIM))
			{
				CV_MAT_ELEM(*sample_all, float, 0, j ) = feat.fCLD[j-EHDDIM*4];
			}
			else if(j<(EHDDIM*4+CLDDIM+BLURDIM))
			{
				CV_MAT_ELEM(*sample_all, float, 0, j ) = feat.fBlur[j-(EHDDIM*4+CLDDIM)];
			}		
			else
				return TEC_INVALID_PARAM;			
		}
		result = imagetypeajudge_svm.predict(sample_all);//使用训练好的模型进行分类

		cvReleaseMat(&sample_all);//释放sample_feature_test
		if(pSobelImg){cvReleaseImage(&pSobelImg);pSobelImg = NULL;}
		if(pGrayImg){cvReleaseImage(&pGrayImg);pGrayImg = NULL;}
		if(pWaterImg){cvReleaseImage(&pWaterImg);pWaterImg = NULL;}		
	}
	if(img_resize_1){cvReleaseImage(&img_resize_1);img_resize_1 = NULL;}
	if(img_roi){cvReleaseImage(&img_roi);img_roi = NULL;}
	if(src_roi){cvReleaseImage(&src_roi);src_roi = NULL;}
	return TOK;
}

//释放资源 
int ReleaseImageTypeAJudgeClassify()
{
	imagetypeajudge_svm.clear();
	return TOK;
}

void version()
{
	printf("....this is image type A judge ........\n");
	printf("............version 2.0.0..............\n");
	printf("......write by Chigo 2012-07-31........\n");
}

}