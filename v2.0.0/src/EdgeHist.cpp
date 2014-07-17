#include <stdio.h>
#include <math.h>
#include <memory.h>
#include "EdgeHist.h"
#include "FILE_MACO.h"

namespace ImageTypeAJudge_2_0_0
{

#define		Te_Define				11
#define		Desired_Num_of_Blocks	 1100 // 1024

#define WIDTH		256
#define HEIGHT	256

//typedef	struct Edge_Histogram_Descriptor
//{
//	double Local_Edge[EHDDIM]; 
//} EHD;

#define	NoEdge						0
#define	vertical_edge				    1
#define	horizontal_edge				2
#define	non_directional_edge		    3
#define	diagonal_45_degree_edge		4
#define	diagonal_135_degree_edge	    5

#define edge_0 0
#define edge_1 1
#define edge_2 2
#define edge_3 3
#define edge_4 4
#define edge_5 5
#define edge_6 6
#define edge_7 7
#define edge_8 8

//EHD	m_pEdge_Histogram;
//char m_pEdge_HistogramElement[80];

double QuantTable[5][8] = { 
	{0.010867,0.057915,0.099526,0.144849,0.195573,0.260504,0.358031,0.530128}, 
	{0.012266,0.069934,0.125879,0.182307,0.243396,0.314563,0.411728,0.564319},
	{0.004193,0.025852,0.046860,0.068519,0.093286,0.123490,0.161505,0.228960},
	{0.004174,0.025924,0.046232,0.067163,0.089655,0.115391,0.151904,0.217745},
	{0.006778,0.051667,0.108650,0.166257,0.224226,0.285691,0.356375,0.450972},
};

unsigned long GetBlockSize(unsigned long image_width,
	unsigned long image_height,
	unsigned long desired_num_of_blocks);

void EdgeHistogramGeneration(unsigned char* pImage_Y,
	unsigned long image_width,
	unsigned long image_height,
	unsigned long block_size,
	double* pLocal_Edge, int Te_Value,unsigned char* pImage_Mask = NULL);

int GetEdgeFeature(unsigned char *pImage_Y,
	int image_width, int block_size,
	int Te_Value,unsigned char* pImage_Mask = NULL);

void StartExtracting(IplImage *MediaData,float *m_pEdge_HistogramElement,IplImage* pMskImg = NULL);

void SetEdgeHistogram(float*	pEdge_Histogram,char *m_pEdge_HistogramElement);

void Make_Global_SemiGlobal(float *LocalHistogramOnly, float *TotalHistogram);

void StartExtracting(IplImage *MediaData,float *m_pEdge_HistogramElement,IplImage* pMskImg)
{
	IplImage *ImageMedia = MediaData;

	unsigned long	desired_num_of_blocks;
	unsigned long	block_size;
	int		Te_Value;
#if 1
	double* pLocal_Edge = NULL;
	pLocal_Edge = new double[EHDDIM];
	memset(pLocal_Edge,0,sizeof(double)*EHDDIM);
#else
	EHD		*pLocal_Edge;
	pLocal_Edge = NULL;
	pLocal_Edge = new	EHD[1];
#endif
	Te_Value = Te_Define;
	desired_num_of_blocks = Desired_Num_of_Blocks;

	//////////////////////// Making Gray Image///////////////
	int i, j, xsize, ysize;
	unsigned char	*pGrayImage = NULL;
	unsigned char *pResampleImage=NULL;
	//add by sprit
	unsigned char *pMaskImage = NULL;
	unsigned char* pResampleMskImg = NULL;

	int max_x = 0, max_y = 0, min_x = ImageMedia->width-1, min_y = ImageMedia->height-1;
	double scale, EWweight, NSweight, EWtop, EWbottom;
	unsigned char NW, NE, SW, SE;
	int min_size, re_xsize, re_ysize;
	xsize = ImageMedia->width;
	ysize = ImageMedia->height;
	min_x = min_y = 0;
	pGrayImage = new unsigned char[xsize*ysize];
	//add by sprit
	if(pMskImg)
		pMaskImage = (unsigned char*)pMskImg->imageData;

	int nSize = xsize * ysize;
//	char *pImgData = ImageMedia->imageData;
	//add by sprit
	unsigned char *pImgData = (unsigned char*)ImageMedia->imageData;
	int iLineBytes = ImageMedia->widthStep;

	if(ImageMedia->nChannels == 3)
	{
		for( j=0; j < ysize; j++)
		{
			uchar b = 0, g = 0, r = 0;
			unsigned char* pOffset = pImgData;
			int steps = j * xsize;
			for( i=0; i < xsize; i++) 
			{
				b = *pOffset;
				pOffset ++;
				g = *pOffset;
				pOffset ++;
				r = *pOffset;
				pOffset ++;
				
			//	pGrayImage[steps+i] = (b + g + r) / 3.0f;
				pGrayImage[steps+i] = 0.212671 * r + 0.715160 * g + 0.072169 * b;
					
			}
			pImgData += iLineBytes;
		}
	}
	else if(ImageMedia->nChannels == 1)
	{
		for( j=0; j < ysize; j++)
		{
			uchar b = 0;
			unsigned char* pOffset = pImgData;
			int steps = j * xsize;
			for( i=0; i < xsize; i++) 
			{
				b = *pOffset;
				pOffset ++;
				pGrayImage[steps+i] = b;
			}
			pImgData += iLineBytes;
		}
	}

	min_size = (xsize>ysize)? ysize: xsize;
	if(min_size<70)
	{
		///////////////////////////////////upsampling///////////////////////////
		scale = 70.0/min_size;
		re_xsize = (int)(xsize*scale+0.5);
		re_ysize = (int)(ysize*scale+0.5);
		pResampleImage = new unsigned char[re_xsize*re_ysize];
		//add by sprit
		if(pMaskImage)
			pResampleMskImg = new unsigned char[re_xsize * re_ysize];

		for(j=0; j<re_ysize; j++)for(i=0; i<re_xsize; i++)
		{
			EWweight = i/scale-floor(i/scale);
			NSweight = j/scale-floor(j/scale);

			NW = pGrayImage[(int)floor(i/scale)+(int)floor(j/scale)*xsize];
			NE = pGrayImage[(int)floor(i/scale)+1+(int)floor(j/scale)*xsize];
			SW = pGrayImage[(int)floor(i/scale)+(int)(floor(j/scale)+1)*xsize];
			SE = pGrayImage[(int)floor(i/scale)+1+(int)(floor(j/scale)+1)*xsize];
			EWtop = NW + EWweight*(NE-NW);
			EWbottom = SW + EWweight*(SE-SW);
			pResampleImage[i+j*re_xsize] = (unsigned char)(EWtop + NSweight*(EWbottom-EWtop)+0.5);

			//add by sprit
			if(pResampleMskImg)
			{
				NW = pMaskImage[(int)floor(i/scale)+(int)floor(j/scale)*xsize];
				NE = pMaskImage[(int)floor(i/scale)+1+(int)floor(j/scale)*xsize];
				SW = pMaskImage[(int)floor(i/scale)+(int)(floor(j/scale)+1)*xsize];
				SE = pMaskImage[(int)floor(i/scale)+1+(int)(floor(j/scale)+1)*xsize];
				EWtop = NW + EWweight*(NE-NW);
				EWbottom = SW + EWweight*(SE-SW);
				pResampleMskImg[i+j*re_xsize] = (unsigned char)(EWtop + NSweight*(EWbottom-EWtop)+0.5);
			}
			
		
		}
		block_size = GetBlockSize(re_xsize, re_ysize, desired_num_of_blocks);
		if(block_size<2)
			block_size = 2;
		EdgeHistogramGeneration(pResampleImage, re_xsize, re_ysize, block_size, pLocal_Edge, Te_Value,pResampleMskImg);
		delete  [] pResampleImage;

		//add by sprit
		delete []pResampleMskImg;
	}
	else
	{
		block_size = GetBlockSize(xsize, ysize, desired_num_of_blocks);
		if(block_size<2)
			block_size = 2;
		EdgeHistogramGeneration(pGrayImage, xsize, ysize, block_size, pLocal_Edge, Te_Value,pMaskImage);
	}

//	SetEdgeHistogram( pLocal_Edge,m_pEdge_HistogramElement );

	for(i = 0; i < EHDDIM; i ++)
	{
#if 1
		m_pEdge_HistogramElement[i] = (float)pLocal_Edge[i];
#else
		m_pEdge_HistogramElement[i] = (float)pLocal_Edge->Local_Edge[i];
#endif
	}

	delete	[] pLocal_Edge;
	delete	[] pGrayImage;
//	return m_pEdge_HistogramElement;
}

//----------------------------------------------------------------------------
unsigned long GetBlockSize(unsigned long image_width, unsigned long image_height, unsigned long desired_num_of_blocks)
{
	unsigned long	block_size;
	double		temp_size;

	temp_size = (double) sqrt((double)(image_width*image_height/desired_num_of_blocks));
	block_size = (unsigned long) (temp_size/2)*2;

	return block_size;
}

//----------------------------------------------------------------------------
void EdgeHistogramGeneration(unsigned char* pImage_Y, 
	unsigned long image_width,
	unsigned long image_height,
	unsigned long block_size,
	double* pLocal_Edge, int Te_Value,unsigned char* pImage_Mask)
{
	int Count_Local[16],sub_local_index;
	int Offset, EdgeTypeOfBlock;
	unsigned int i, j;
	long	LongTyp_Local_Edge[EHDDIM];

	// Clear
	memset(Count_Local, 0, 16*sizeof(int));		
	memset(LongTyp_Local_Edge, 0, EHDDIM*sizeof(long));

	for(j=0; j<=image_height-block_size; j+=block_size)
		for(i=0; i<=image_width-block_size; i+=block_size)
		{
			sub_local_index = (int)(i*4/image_width)+(int)(j*4/image_height)*4;
			//ori
			Count_Local[sub_local_index]++;

			Offset = image_width*j+i;
			//add by sprit
	/*		if(pImage_Mask != NULL)
			{
				for(int m = 0; m < block_size; m ++)
				{
					int steps = m * block_size;
					for(int n = 0; n < block_size; n ++)
					{
						unsigned char* pMskData = pImage_Mask+Offset;
						if(pMskData[steps+n]) Count_Local[sub_local_index]++;
					}
				}
			}
			else Count_Local[sub_local_index]++;*/


			EdgeTypeOfBlock = GetEdgeFeature(pImage_Y+Offset, image_width,
				block_size, Te_Value,pImage_Mask);
			switch(EdgeTypeOfBlock) 
			{
#if 0
			case edge_0:
				LongTyp_Local_Edge[sub_local_index*9]++;
				break;
			case edge_1:
				LongTyp_Local_Edge[sub_local_index*9+1]++;
				break;
			case edge_2:
				LongTyp_Local_Edge[sub_local_index*9+2]++;
				break;
			case edge_3:
				LongTyp_Local_Edge[sub_local_index*9+3]++;
				break;
			case edge_4:
				LongTyp_Local_Edge[sub_local_index*9+4]++;
				break;
			case edge_5:
				LongTyp_Local_Edge[sub_local_index*9+5]++;
				break;
			case edge_6:
				LongTyp_Local_Edge[sub_local_index*9+6]++;
				break;
			case edge_7:
				LongTyp_Local_Edge[sub_local_index*9+7]++;
				break;
			case edge_8:
				LongTyp_Local_Edge[sub_local_index*9+8]++;
				break;
#else
				case NoEdge:
					break;
				case vertical_edge:
					LongTyp_Local_Edge[sub_local_index*5]++;
					break;
				case horizontal_edge:
					LongTyp_Local_Edge[sub_local_index*5+1]++;
					break;
				case diagonal_45_degree_edge:
					LongTyp_Local_Edge[sub_local_index*5+2]++;
					break;
				case diagonal_135_degree_edge:
					LongTyp_Local_Edge[sub_local_index*5+3]++;
					break;
				case non_directional_edge:
					LongTyp_Local_Edge[sub_local_index*5+4]++;
					break;
#endif
			} //switch(EdgeTypeOfBlock)
		} // for( i )

		for( i=0; i<EHDDIM; i++) 
		{			// Range 0.0 ~ 1.0
			sub_local_index = (int)(i/5);
#if 1
			if(Count_Local[sub_local_index] == 0)
				pLocal_Edge[i] = 0;
			else
				pLocal_Edge[i] = (double)LongTyp_Local_Edge[i]/(Count_Local[sub_local_index]);
#else
			if(Count_Local[sub_local_index] == 0)
				pLocal_Edge->Local_Edge[i] = 0;
			else
				pLocal_Edge->Local_Edge[i] =
					(double)LongTyp_Local_Edge[i]/(Count_Local[sub_local_index]);
#endif
		}
}

//----------------------------------------------------------------------------------------------------------------------
int GetEdgeFeature(unsigned char *pImage_Y,
	int image_width,
	int block_size,
	int Te_Value,unsigned char* pImage_Mask)
{
	int i, j;
	double	d1, d2, d3, d4;
	int e_index;
	double dc_th = Te_Value;
	double e_h, e_v, e_45, e_135, e_m, e_max;

	d1=0.0;
	d2=0.0;
	d3=0.0;
	d4=0.0;

	for(j=0; j<block_size; j++)for(i=0; i<block_size; i++){

		if(pImage_Mask != NULL && pImage_Mask[i+image_width*j] == 0) continue;

		if(j<block_size/2)
		{
			if(i<block_size/2)
				d1+=(pImage_Y[i+image_width*j]);
			else
				d2+=(pImage_Y[i+image_width*j]);
		}
		else
		{
			if(i<block_size/2)
				d3+=(pImage_Y[i+image_width*j]);
			else
				d4+=(pImage_Y[i+image_width*j]);
		}
	}
	d1 = d1/(block_size*block_size/4.0);
	d2 = d2/(block_size*block_size/4.0);
	d3 = d3/(block_size*block_size/4.0);
	d4 = d4/(block_size*block_size/4.0);

#if 0
	//add by sprit
	float dx = (float)d1 - (float)d2;
	float dy = (float)d3 - (float)d4;

	float theta = atan2(dy, dx);
	if (theta < 0)	theta = (float)(theta+CV_PI);	// normalize to [0, PI], CV_PI
	if (theta >= CV_PI)	theta = 0.0f;
	theta = (float)(theta*180/CV_PI);
	e_index = theta / 20;
#else
	e_h = fabs(d1+d2-(d3+d4));
	e_v = fabs(d1+d3-(d2+d4));
	e_45 = sqrt((float)2)*fabs(d1-d4);
	e_135 = sqrt((float)2)*fabs(d2-d3);

	e_m = 2*fabs(d1-d2-d3+d4);

	e_max = e_v;
	e_index = vertical_edge;
	if(e_h>e_max)
	{
		e_max=e_h;
		e_index = horizontal_edge;
	}
	if(e_45>e_max)
	{
		e_max=e_45;
		e_index = diagonal_45_degree_edge;
	}
	if(e_135>e_max)
	{
		e_max=e_135;
		e_index = diagonal_135_degree_edge;
	}
	if(e_m>e_max)
	{
		e_max=e_m;
		e_index = non_directional_edge;
	}
	if(e_max<dc_th)
		e_index = NoEdge;
#endif
	return(e_index);
}

void SetEdgeHistogram(float*	pEdge_Histogram, char* m_pEdge_HistogramElement)
{
	int i, j;
	double iQuantValue;

	for( i=0; i < EHDDIM; i++ ) 
	{
		j=0;
		while(1)
		{
			if( j < 7 ) // SIZE-1 
				iQuantValue = (QuantTable[i%5][j]+QuantTable[i%5][j+1])/2.0;
			else 
				iQuantValue = 1.0;
#if 1
			if(pEdge_Histogram[i] <= iQuantValue)
#else
			if(pEdge_Histogram->Local_Edge[i] <= iQuantValue)
#endif
				break;
			j++;
		}
		m_pEdge_HistogramElement[i] = j;
	}
	/*
	for( i=0; i < 80; i++ )
	{
		m_pEdge_Histogram.Local_Edge[i] = QuantTable[ i%5 ][ m_pEdge_HistogramElement[i] ];
	}
	*/
}

void EdgeHistExtractor(IplImage *Image, float fEHD[],IplImage* pMskImg)
{
	if(!fEHD ||!Image)
	{
		printf("Please CHECK your input image!\n");

		return;
	}

	IplImage *ImageMedia = cvCreateImage(cvSize(WIDTH, HEIGHT), Image->depth, Image->nChannels);
	cvResize(Image, ImageMedia);

	//ADD BY SPRIT
	IplImage *ImageMask = NULL;
	if(pMskImg) 
	{
		ImageMask = cvCreateImage(cvSize(WIDTH, HEIGHT), pMskImg->depth, pMskImg->nChannels);
		cvResize(pMskImg,ImageMask);
		cvThreshold(ImageMask,ImageMask,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
	}

	float m_pEdge_HistogramElement[EHDDIM];
	StartExtracting(ImageMedia,m_pEdge_HistogramElement,ImageMask);
	

	float *pEHD = m_pEdge_HistogramElement; 

	for(int i = 0; i < EHDDIM; i++)
	{
		fEHD[i] = pEHD[i];
	}


	cvReleaseImage(&ImageMedia);
	cvReleaseImage(&ImageMask);
	return;
}

//----------------------------------------------------------------------------
void Make_Global_SemiGlobal(float *LocalHistogramOnly, float *TotalHistogram)
//void Make_Global_SemiGlobal(double *LocalHistogramOnly, double *TotalHistogram)
{
	int i, j;	
	memcpy(TotalHistogram+5, LocalHistogramOnly, 80*sizeof(float) );
// Make Global Histogram Start
	for(i=0; i<5; i++)
	  TotalHistogram[i]=0.0;
	for( j=0; j < 80; j+=5) 
	{
		for( i=0; i < 5; i++) 
		{
			TotalHistogram[i] += TotalHistogram[5+i+j]; 
		}
	}  // for( j ) 
	for(i=0; i<5; i++)
// Global *5.
		TotalHistogram[i] = TotalHistogram[i]*5/16.0;
// Make Global Histogram end

// Make Semi-Global Histogram start
	for(i=85; i <105; i++) 
	{
		j = i-85;
		TotalHistogram[i] =
			(TotalHistogram[5+j]
			+TotalHistogram[5+20+j]
			+TotalHistogram[5+40+j]
			+TotalHistogram[5+60+j])/4.0;
	}
	for(i=105; i < 125; i++) 
	{
		j = i-105;
		TotalHistogram[i] =
			(TotalHistogram[5+20*(j/5)+j%5]
			+TotalHistogram[5+20*(j/5)+j%5+5]
			+TotalHistogram[5+20*(j/5)+j%5+10]
			+TotalHistogram[5+20*(j/5)+j%5+15])/4.0;
	}
///////////////////////////////////////////////////////
//				4 area Semi-Global
///////////////////////////////////////////////////////
//  Upper area 2.
	for(i=125; i < 135; i++) 
	{
		j = i-125;    // j = 0 ~ 9
		TotalHistogram[i] =
			(TotalHistogram[5+10*(j/5)+0+j%5]
				   +TotalHistogram[5+10*(j/5)+5+j%5]
			       +TotalHistogram[5+10*(j/5)+20+j%5]
			       +TotalHistogram[5+10*(j/5)+25+j%5])/4.0;
	}
//  Down area 2.
	for(i=135; i < 145; i++) 
	{
		j = i-135;    // j = 0 ~ 9
		TotalHistogram[i] =
			(TotalHistogram[5+10*(j/5)+40+j%5]
			       +TotalHistogram[5+10*(j/5)+45+j%5]
			       +TotalHistogram[5+10*(j/5)+60+j%5]
			       +TotalHistogram[5+10*(j/5)+65+j%5])/4.0;
	}
// Center Area 1 
	for(i=145; i < 150; i++) 
	{
		j = i-145;    // j = 0 ~ 9
		TotalHistogram[i] =
			(TotalHistogram[5+25+j%5]
			       +TotalHistogram[5+30+j%5]
			       +TotalHistogram[5+45+j%5]
			       +TotalHistogram[5+50+j%5])/4.0;
	}
// Make Semi-Global Histogram end
	
}

double EHDDist(int EHD1[], int EHD2[])
{
	// ------------------------- Calculate the distance ------------------------
	double EHD1_LOCAL[80], EHD2_LOCAL[80];
	double EHD1_ALL[150], EHD2_ALL[150];
	for(int i =0; i < 80; i++)
	{
		EHD1_LOCAL[i] = (double)EHD1[i];
		EHD2_LOCAL[i] = (double)EHD2[i];
	}
	
//	Make_Global_SemiGlobal(EHD1_LOCAL, EHD1_ALL);
//	Make_Global_SemiGlobal(EHD2_LOCAL, EHD2_ALL);
	
	double dist = 0.0;
	for(int i=0; i < 80+70; i++)
	{
	  // Global(5)+Semi_Global(65) 
	  dist += (fabs((EHD1_ALL[i] - EHD2_ALL[i])));
	}

	return 	dist;
};

}