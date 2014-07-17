#pragma once
#include <cv.h>

namespace ImageTypeAJudge_2_0_0
{
int getSobel(IplImage* pSrcImg,IplImage* pSobelImg);

IplImage* ImageQuality_FFT(IplImage* pSrcImg);

int ImageQuality_FFT2(IplImage* pSrcImg,IplImage* pDstImg);

int ImageQuality_IFFT(IplImage* pSrcImg,IplImage* pDstImg);

bool EdgeDet_Soble(const IplImage * pOriData, IplImage * pEdgeData);

int GetEdgeImg(IplImage* pSrcImg, IplImage* pSobelImg);

void quickAdaptiveThreshold(unsigned char* grayscale, unsigned char*& thres, int width, int height ); 

int directionality(IplImage* pSrcImg,IplImage* pData);

int contrast(IplImage* pSrcImg, IplImage* pData);

int coarseness(IplImage* pSrcImg,IplImage* pData);
}
