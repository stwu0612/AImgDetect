#include <math.h>
#include <iostream>
#include "ColorLayout.h"
#include "FILE_MACO.h"
#include <fstream>
using namespace std;

namespace ImageTypeAJudge_2_0_0
{

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#ifndef FLT_MAX
#define FLT_MAX 10000000000.0
#endif

unsigned char zigzag_scan[64]={        /* Zig-Zag scan pattern  */
	0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,
	12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
	35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
	58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63
};

#define NUMBEROFYCOEFF 6
#define NUMBEROFCCOEFF 3

#define WIDTH	256
#define HEIGHT	256

double c[8][8]; /* transform coefficients */

void CreateSmallImage(IplImage *src, int small_img[3][64]);

void init_fdct();

inline void fdct(int *block);

inline int quant_ydc(int i);

inline int quant_cdc(int i);

inline int quant_ac(int i);

void GetBGRChannelData(IplImage *img, unsigned char *B, unsigned char *G, unsigned char *R)
{
	char *pImgData = img->imageData;

	int nImgSize = img->width*img->height;
	int counter = 0;
	for(int i = 0; i < nImgSize; i++)
	{
		char b = 0, g = 0, r = 0;

		b = pImgData[counter];
		g = pImgData[counter+1];
		r = pImgData[counter+2];

		B[i] = (unsigned char)b;
		G[i] = (unsigned char)g;
		R[i] = (unsigned char)r;

		counter += 3;
	}

	return;
}

void ColorLayoutExtractor(IplImage *Image, float fCLD[])
{
	if(!Image)	return;
	if(Image->nChannels != 3) {memset(fCLD,0,sizeof(float)*CLDDIM);return ;}

	IplImage *ImageMedia = cvCreateImage(cvSize(WIDTH, HEIGHT), Image->depth, Image->nChannels);
	cvResize(Image, ImageMedia);

	// Descriptor data
	int NumberOfYCoeff = NUMBEROFYCOEFF;
	int NumberOfCCoeff = NUMBEROFCCOEFF;
	if(NumberOfYCoeff<1) NumberOfYCoeff = 1;
	else if(NumberOfYCoeff > 64) NumberOfYCoeff = 64;
	if(NumberOfCCoeff < 1) NumberOfCCoeff = 1;
	else if(NumberOfCCoeff > 64) NumberOfCCoeff = 64;

	init_fdct();

	int small_img[3][64];
	CreateSmallImage(ImageMedia, small_img);
	

	fdct(small_img[0]);
	fdct(small_img[1]);
	fdct(small_img[2]);
	
	int YCoeff[64], CbCoeff[64], CrCoeff[64];
	YCoeff[0]=quant_ydc(small_img[0][0]/8)>>1;
	CbCoeff[0]=quant_cdc(small_img[1][0]/8);
	CrCoeff[0]=quant_cdc(small_img[2][0]/8);

	for(int i=1;i<64;i++)
	{
		YCoeff[i]=quant_ac((small_img[0][(zigzag_scan[i])])/2)>>3;
		CbCoeff[i]=quant_ac(small_img[1][(zigzag_scan[i])])>>3;
		CrCoeff[i]=quant_ac(small_img[2][(zigzag_scan[i])])>>3;
	}

	int cldCounter = 0;
	int tmpMax = 0;
	tmpMax = 0;
	for(int i = 0; i < 6; i++)
	{
		fCLD[cldCounter++] = YCoeff[i];
		if(fCLD[cldCounter-1] > tmpMax)
			tmpMax = fCLD[cldCounter-1];
	}
	for(int i = 0; i < 3; i++) //3
	{
		fCLD[cldCounter++] = CbCoeff[i];
		if(fCLD[cldCounter-1] > tmpMax)
			tmpMax = fCLD[cldCounter-1];
	}
	for(int i = 0; i < 3; i++) //3
	{
		fCLD[cldCounter++] = CrCoeff[i];
		if(fCLD[cldCounter-1] > tmpMax)
			tmpMax = fCLD[cldCounter-1];
	}

	for(int i = 0; i < CLDDIM; i ++)
	{
		fCLD[i] /= (float)(tmpMax + 0.0l);
	}
	cvReleaseImage(&ImageMedia);
	
	return;
}

// 8x8 DCT 
// init_fdct and fdct are developed by MPEG Software Simulation Group
void init_fdct()
{
	//#define COMPUTE_COS
	int i, j;
#ifdef COMPUTE_COS
	double s;
#else // for memory debugging!!
	double d_cos [8][8] =
	{{3.535534e-01, 3.535534e-01, 3.535534e-01, 3.535534e-01,
	3.535534e-01, 3.535534e-01, 3.535534e-01, 3.535534e-01},
	{4.903926e-01, 4.157348e-01, 2.777851e-01, 9.754516e-02,
	-9.754516e-02, -2.777851e-01, -4.157348e-01, -4.903926e-01},
	{4.619398e-01, 1.913417e-01, -1.913417e-01, -4.619398e-01,
	-4.619398e-01, -1.913417e-01, 1.913417e-01, 4.619398e-01},
	{4.157348e-01, -9.754516e-02, -4.903926e-01, -2.777851e-01,
	2.777851e-01, 4.903926e-01, 9.754516e-02, -4.157348e-01},
	{3.535534e-01, -3.535534e-01, -3.535534e-01, 3.535534e-01,
	3.535534e-01, -3.535534e-01, -3.535534e-01, 3.535534e-01},
	{2.777851e-01, -4.903926e-01, 9.754516e-02, 4.157348e-01,
	-4.157348e-01, -9.754516e-02, 4.903926e-01, -2.777851e-01},
	{1.913417e-01, -4.619398e-01, 4.619398e-01, -1.913417e-01,
	-1.913417e-01, 4.619398e-01, -4.619398e-01, 1.913417e-01},
	{9.754516e-02, -2.777851e-01, 4.157348e-01, -4.903926e-01,
	4.903926e-01, -4.157348e-01, 2.777851e-01, -9.754516e-02}};
#endif

	for (i=0; i<8; i++){
#ifdef COMPUTE_COS
		s = (i==0) ? sqrt(0.125) : 0.5;
#endif
		for (j=0; j<8; j++) {
#ifdef COMPUTE_COS
			c[i][j] = s * cos((M_PI/8.0)*i*(j+0.5));
			fprintf(stderr,"%le ", c[i][j]);
#else
			c[i][j] = d_cos [i][j];

#endif
		}
	}
}

//----------------------------------------------------------------------------
void fdct(int *block)
{
	int i, j, k;
	double s;
	double tmp[64];
	for (i=0; i<8; i++){
		for (j=0; j<8; j++){
			s = 0.0;
			for (k=0; k<8; k++) s += c[j][k] * block[8*i+k];
			tmp[8*i+j] = s;
		}
	}
	for (j=0; j<8; j++){
		for (i=0; i<8; i++){
			s = 0.0;
			for (k=0; k<8; k++) s += c[i][k] * tmp[8*k+j];
			block[8*i+j] = (int)floor(s+0.499999);
		}
	}
}

//改进cld方式 每块求的不是平均值，求的是主颜色，利用hsv颜色空间量化方式求的
//参考文献<基于MPEG7颜色纹理特征图像检索技术---作者华南理工赵莉>
int GetSubRectMainValue(IplImage* pSrcImg,int YCrCb[3][64])
{
	//IplImage* pHSVImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,pSrcImg->nChannels);
	//IplImage* pYCrCbImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,pSrcImg->nChannels);
	//cvCvtColor(pSrcImg,pHSVImg,CV_BGR2HSV);
	//cvCvtColor(pSrcImg,pYCrCbImg,CV_BGR2YCrCb);
	//IplImage* pH = cvCreateImage(cvGetSize(pSrcImg),8,1);
	//IplImage* pS = cvCreateImage(cvGetSize(pSrcImg),8,1);
	//IplImage* pV = cvCreateImage(cvGetSize(pSrcImg),8,1);
	//cvSplit(pHSVImg,pH,pS,pV,0);
	int nWid = pSrcImg->width;
	int nHei = pSrcImg->height;

	int numSub[64];
	int pixelNum[64];
	int pixelValue[3][64];
	const int color_level_dim = 64;
	int vHist[64][color_level_dim]; int yValue[64][color_level_dim]; 
	int cValue[64][color_level_dim]; int bValue[64][color_level_dim];
	for(int i = 0; i < 64; i ++)
	{
		numSub[i] = 0;
		pixelNum[i] = 0;
		pixelValue[0][i] = 0;
		pixelValue[1][i] = 0;
		pixelValue[2][i] = 0;

		YCrCb[0][i] = 0;
		YCrCb[1][i] = 0;
		YCrCb[2][i] = 0;
		for(int j = 0; j < color_level_dim; j ++)
		{
			vHist[i][j] = 0;
			yValue[i][j] = 0;
			cValue[i][j] = 0;
			bValue[i][j] = 0;
		}
	}

	for(int i = 0; i < nHei; i ++)
	{
		for(int j = 0; j < nWid; j ++)
		{
			int y_axis = (int)(i/(nHei/8.0));
			int x_axis = (int)(j/(nWid/8.0));
			int k = y_axis * 8 + x_axis;

		/*	uchar h = CV_IMAGE_ELEM(pH,uchar,i,j);
			uchar s = CV_IMAGE_ELEM(pS,uchar,i,j);
			uchar v = CV_IMAGE_ELEM(pV,uchar,i,j);*/

			uchar B = CV_IMAGE_ELEM(pSrcImg,uchar,i,j*3);
			uchar G = CV_IMAGE_ELEM(pSrcImg,uchar,i,j*3+1);
			uchar R = CV_IMAGE_ELEM(pSrcImg,uchar,i,j*3+2);

			uchar b = B >> 6;
			uchar g = G >> 6;
			uchar r = R >> 6;

			double yy = ( 0.299*R + 0.587*G + 0.114*B)/256.0; 
			

			uchar ch = r*16+g*4+b;
			/*if(v < 51)
			{
				ch = 0;
			}
			else if(s < 51 && v >= 51) 
			{
				ch = ceil((v / 255.0 -0.2)*7/0.8);
			}
			else
			{
				s = s < 166 ? 0 : 1;
				v = v < 179 ? 0 : 1;
				
				if(h < 16) h = 0;
				else if(h < 29) h = 1;
				else if(h < 50) h = 2;
				else if(h < 110)h = 3;
				else if(h < 132)h = 4;
				else if(h < 197)h = 5;
				else if(h < 234)h = 6;
				else if(h < 255)h = 0;

				ch = 4*h+2*s+v+8;
				
			}*/

			numSub[k] ++;
			vHist[k][ch] ++;

			yValue[k][ch] += (int)(219.0 * yy + 16.5); // Y
			bValue[k][ch] += (int)(224.0 * 0.564 * (B/256.0*1.0 - yy) + 128.5); // Cb
			cValue[k][ch] += (int)(224.0 * 0.713 * (R/256.0*1.0 - yy) + 128.5); // Cr
		}
	}

	int tmpMax[64];
	int rtmpMax[64];
	int index[64];
	int rindex[64];
	
	vector<pair<int,int> > pVect[64];
	pair<int,int> tmp_pair;

	for(int j = 0; j < 64; j ++)
	{
		pVect[j].clear();
		for(int i = 0; i < color_level_dim; i ++)
		{
			tmp_pair.first = vHist[j][i];
			tmp_pair.second = i;
			pVect[j].push_back(tmp_pair);
		}
		sort(pVect[j].begin(),pVect[j].end());
	}
	
	double ratoi = 0;
		
	for(int j = 0; j < 64; j ++)
	{
		for(int i = color_level_dim-1; i >= color_level_dim-7; i --)
		{
		//	if(vHist[j][i] * 1.0 / numSub[j] > 0.1)
		//	{
				pixelNum[j] += pVect[j][i].first;
				pixelValue[0][j] += yValue[j][pVect[j][i].second];
				pixelValue[1][j] += bValue[j][pVect[j][i].second];
				pixelValue[2][j] += cValue[j][pVect[j][i].second];
		//	}
		}

		if(pixelNum[j] != 0)
		{
			ratoi = 1.0 / pixelNum[j];
			YCrCb[0][j] =  pixelValue[0][j] * ratoi;
			YCrCb[1][j] =  pixelValue[1][j] * ratoi;
			YCrCb[2][j] =  pixelValue[2][j] * ratoi;
		}

	}

	//if(pHSVImg){cvReleaseImage(&pHSVImg);pHSVImg = NULL;}
	//if(pH){cvReleaseImage(&pH);pH = NULL;}
	//if(pV){cvReleaseImage(&pV);pV = NULL;}
	//if(pS){cvReleaseImage(&pS);pS = NULL;}
	//if(pYCrCbImg){cvReleaseImage(&pYCrCbImg);pYCrCbImg = NULL;}
	return 0;
}

//
//  End of routines those are developed by MPEG Software Simulation Group
//

//----------------------------------------------------------------------------
void CreateSmallImage(IplImage *src, int small_img[3][64])
{
	int y_axis, x_axis;
	int i, j, k ;
	int x, y;
	int small_block_sum[3][64];
	int cnt[64];

	for(i = 0; i < (8 * 8) ; i++){
		cnt[i]=0;
		for(j=0;j<3;j++){
			small_block_sum[j][i]=0;
			small_img[j][i] = 0;
		}
	}
//ori
#if 1
	int nSize = src->width * src->height;
	unsigned char *pChannelB = new unsigned char[nSize];
	unsigned char *pChannelG = new unsigned char[nSize];
	unsigned char *pChannelR = new unsigned char[nSize];
	
	GetBGRChannelData(src, pChannelB, pChannelG, pChannelR);

	unsigned char *pB = pChannelB;
	unsigned char *pG = pChannelG;
	unsigned char *pR = pChannelR;

	int R, G, B;
	double yy = 0;
	int nWid = src->width; int nHei = src->height;
	for(y=0; y<nHei; y++)
	{
		for(x=0; x<nWid; x++)
		{
			y_axis = (int)(y/(nHei/8.0));
			x_axis = (int)(x/(nWid/8.0));
			k = y_axis * 8 + x_axis;

			G = *pG++;
			B = *pB++;
			R = *pR++;

			// RGB to YCbCr
			yy = ( 0.299*R + 0.587*G + 0.114*B)/256.0; 
			small_block_sum[0][k] += (int)(219.0 * yy + 16.5); // Y
			small_block_sum[1][k] += (int)(224.0 * 0.564 * (B/256.0*1.0 - yy) + 128.5); // Cb
			small_block_sum[2][k] += (int)(224.0 * 0.713 * (R/256.0*1.0 - yy) + 128.5); // Cr

			cnt[k]++;
		}
	}
	
#else	
	//add by sprit2011-11-18
	GetSubRectMainValue(src,small_block_sum);
#endif
	// create 8x8 image
	for(i=0; i<8; i++){
		for(j=0; j<8; j++){
			for(k=0; k<3; k++){
				//ori
#if 1
				if(cnt[i*8+j]) 
					small_img[k][i*8+j] = (small_block_sum[k][i*8+j] / cnt[i*8+j]);
				else 
					small_img[k][i*8+j] = 0;
#else
				//add by sprit 2011-11-18
				small_img[k][i*8+j] = small_block_sum[k][i*8+j];
#endif
			}
		}
	}

#if 1
	delete [] pChannelB;
	delete [] pChannelG;
	delete [] pChannelR;
#endif
}

//----------------------------------------------------------------------------
int quant_ydc(int i)
{
	int j;
	if(i>191) j=112+(i-192)/4;
	else if(i>159) j=96+(i-160)/2;
	else if(i>95) j=32+(i-96);
	else if(i>63) j=16+(i-64)/2;
	else j=i/4;
	return j;
}

//----------------------------------------------------------------------------
int quant_cdc(int i)
{
	int j;
	if(i>191) j=63;
	else if(i>159) j=56+(i-160)/4;
	else if(i>143) j=48+(i-144)/2;
	else if(i>111) j=16+(i-112);
	else if(i>95) j=8+(i-96)/2;
	else if(i>63) j=(i-64)/4;
	else j=0;
	return j;
}

//----------------------------------------------------------------------------
int quant_ac(int i)
{
	int j;
	if(i>239) i= 239;
	if(i<-256) i= -256;
	if ((abs(i)) > 127) j= 64 + (abs(i))/4;
	else if ((abs(i)) > 63) j=32+(abs(i))/2;
	else j=abs(i);
	j = (i<0)?-j:j;
	j+=132;
	return j;
}

//-----------------------------------------------------------------------------
double CLDDist(int CLD1[], int CLD2[])
{
	double DY = sqrt((double)
					(2*(CLD1[0] - CLD2[0])*(CLD1[0] - CLD2[0]) + 
					 2*(CLD1[1] - CLD2[1])*(CLD1[1] - CLD2[1]) + 
					 2*(CLD1[2] - CLD2[2])*(CLD1[2] - CLD2[2]) +
					 1*(CLD1[3] - CLD2[3])*(CLD1[3] - CLD2[3]) + 
					 1*(CLD1[4] - CLD2[4])*(CLD1[4] - CLD2[4]) + 
					 1*(CLD1[5] - CLD2[5])*(CLD1[5] - CLD2[5]))
					 );

	double DCb = sqrt((double)
					(2*(CLD1[6] - CLD2[6])*(CLD1[6] - CLD2[6]) + 
					 1*(CLD1[7] - CLD2[7])*(CLD1[7] - CLD2[7]) + 
					 1*(CLD1[8] - CLD2[8])*(CLD1[8] - CLD2[8]))
					 );

	double DCr = sqrt((double)
					(4*(CLD1[9] - CLD2[9])*(CLD1[9] - CLD2[9]) + 
					 2*(CLD1[10] - CLD2[10])*(CLD1[10] - CLD2[10]) + 
					 2*(CLD1[11] - CLD2[11])*(CLD1[11] - CLD2[11]))
					 );

	double D = DY + DCb + DCr;

	return D;
};
}