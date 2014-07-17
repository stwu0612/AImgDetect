#include "Process.h"
#include "math.h"
#include <vector>

#include <cxcore.h>
#include <cv.h>
#include <ml.h>

#include <fstream>
using namespace std;

namespace ImageTypeAJudge_2_0_0
{

int GetEdgeImg(IplImage* pSrcImg, IplImage* pSobelImg)
{
	if(pSobelImg->nChannels != 1 || pSrcImg->nChannels != 1)
		return -1;
	int nWid = pSrcImg->width; int nHei = pSrcImg->height;

	cvZero(pSobelImg);

	uchar* pData = (uchar*)pSrcImg->imageData;
	uchar* pSobelData = (uchar*)pSobelImg->imageData;

	int iLineBytes = pSrcImg->widthStep;
	int iSobelBytes = pSobelImg->widthStep;

	for(int i = 1; i < nHei-1; i ++)
	{
		uchar* pOffset = pData;
		uchar* pSobelOff = pSobelData;
		for(int j = 1; j < nWid-1; j ++)
		{
			uchar abs1 = abs(*pOffset - *(pOffset+1));
			uchar abs2 = abs(*pOffset - *(pOffset+2));

			*pSobelOff = max(abs1,abs2);

			pOffset ++;
			pSobelOff ++;
		}
		pData += iLineBytes;
		pSobelData += iSobelBytes;
	}

	return 0;
}

bool EdgeDet_Soble(const IplImage * pOriData, IplImage * pEdgeData)
{
	int X, Y;
	int I, J;
	int sumX,sumY,SUM;
	int iWidth = pOriData->width;
	int iHeight = pOriData->height;

	if(pOriData->nChannels != 1 || pEdgeData->nChannels != 1)
		return false;
	uchar pixel = 0;
	//	memset(pEdgeData,0,sizeof(uchar)*iWidth*iHeight);
	cvZero(pEdgeData);

	/* 3x3 Sobel mask.  Ref:  www.cee.hw.ac.uk/hipr/html/sobel.html */
	int SOBEL_X[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
	int SOBEL_Y[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

	for(Y = 0; Y < iHeight; Y++) 
	{
		for(X = 0; X < iWidth; X++) 
		{
			sumX = 0;
			sumY = 0;

			/* image boundaries */
			if(Y == 0 || Y == iHeight - 1)
				SUM = 0;
			else if(X == 0 || X == iWidth - 1)
				SUM = 0;

			/* Convolution starts here */
			else   
			{
				/*-------X GRADIENT APPROXIMATION------*/
				for(I = -1; I <= 1; I++)  {
					for(J = -1; J <= 1; J++)  {
						//	sumX = sumX + (int)(pOriData[(Y+J) * iWidth + (X+I)] * SOBEL_X[I+1][J+1]);
						sumX = sumX + (int)(((uchar*)(pOriData->imageData + (Y + J) * pOriData->widthStep))[X + I] * SOBEL_X[I+1][J+1]);
					}
				}

				/*-------Y GRADIENT APPROXIMATION-------*/
				for(I = -1; I <= 1; I++)  {
					for(J = -1; J <= 1; J++)  {
						//	sumY = sumY + (int)(pOriData[(Y+J) * iWidth + (X+I)] * SOBEL_Y[I+1][J+1]);
						sumY = sumY + (int)(((uchar*)(pOriData->imageData + (Y + J) * pOriData->widthStep))[X + I] * SOBEL_Y[I+1][J+1]);
					}
				}


				//GRADIENT MAGNITUDE APPROXIMATION (Myler p.218)	
				SUM = abs(sumX) + abs(sumY); 
				//SUM = sqrt(sumX * sumX + sumY * sumY);

				if(SUM > 255) SUM = 255; 
				if(SUM < 0) SUM = 0;
				//	if(SUM >= 100) SUM = 255;
				//	else SUM = 0;
			}
			((uchar*)(pEdgeData->imageData + Y * pEdgeData->widthStep))[X] = (uchar)SUM;
			//	pEdgeData[Y * iWidth + X] = SUM;
		}
	}
	return true;
}

void cvShiftDFT(CvArr * src_arr, CvArr * dst_arr )
{
	CvMat * tmp = NULL;
	CvMat q1stub, q2stub;
	CvMat q3stub, q4stub;
	CvMat d1stub, d2stub;
	CvMat d3stub, d4stub;
	CvMat * q1, * q2, * q3, * q4;
	CvMat * d1, * d2, * d3, * d4;

	CvSize size = cvGetSize(src_arr);
	CvSize dst_size = cvGetSize(dst_arr);
	int cx, cy;

	if(dst_size.width != size.width || 
		dst_size.height != size.height)
	{
		cvError( CV_StsUnmatchedSizes, "cvShiftDFT", "Source and Destination arrays must have equal sizes", __FILE__, __LINE__ );   
	}

	if(src_arr == dst_arr)	////
	{
		tmp = cvCreateMat(size.height/2, size.width/2, cvGetElemType(src_arr));
	}

	cx = size.width/2;
	cy = size.height/2; // image center

	q1 = cvGetSubRect( src_arr, &q1stub, cvRect(0,0,cx, cy) );	// 2 quadrant
	q2 = cvGetSubRect( src_arr, &q2stub, cvRect(cx,0,cx,cy) );	// 1 quadrant
	q3 = cvGetSubRect( src_arr, &q3stub, cvRect(cx,cy,cx,cy) );	// 4 quadrant
	q4 = cvGetSubRect( src_arr, &q4stub, cvRect(0,cy,cx,cy) );	// 3 quadrant
	d1 = cvGetSubRect( src_arr, &d1stub, cvRect(0,0,cx,cy) );
	d2 = cvGetSubRect( src_arr, &d2stub, cvRect(cx,0,cx,cy) );
	d3 = cvGetSubRect( src_arr, &d3stub, cvRect(cx,cy,cx,cy) );
	d4 = cvGetSubRect( src_arr, &d4stub, cvRect(0,cy,cx,cy) );

	if(src_arr != dst_arr)
	{
		if( !CV_ARE_TYPES_EQ( q1, d1 ))
		{
			cvError( CV_StsUnmatchedFormats, "cvShiftDFT", "Source and Destination arrays must have the same format", __FILE__, __LINE__ ); 
		}
		cvCopy(q3, d1, 0);
		cvCopy(q4, d2, 0);
		cvCopy(q1, d3, 0);
		cvCopy(q2, d4, 0);
	}
	else
	{
		cvCopy(q3, tmp, 0);
		cvCopy(q1, q3, 0);
		cvCopy(tmp, q1, 0);
		cvCopy(q4, tmp, 0);
		cvCopy(q2, q4, 0);
		cvCopy(tmp, q2, 0);
	}

	if(tmp){cvReleaseMat(&tmp);tmp = NULL;}
}



IplImage* ImageQuality_FFT(IplImage* pSrcImg)
{
	IplImage * im = NULL;

	IplImage * realInput = NULL;
	IplImage * imaginaryInput = NULL;
	IplImage * complexInput = NULL;
	int dft_M, dft_N;

	CvMat* dft_A = NULL;
	CvMat tmp;
	IplImage * image_Re = NULL;
	IplImage * image_Im = NULL;
	double m, M;

	
	im = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
	imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
	complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);


	if(pSrcImg->nChannels == 3) cvCvtColor(pSrcImg,im,CV_BGR2GRAY);
	else cvCopy(pSrcImg,im);

//	cvSmooth(im,im,CV_GAUSSIAN);

	cvScale(im, realInput, 1.0, 0.0);	// change depth to 64F
	cvZero(imaginaryInput);
	cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);

	dft_M = cvGetOptimalDFTSize( im->height);
	dft_N = cvGetOptimalDFTSize( im->width);

	dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 );
	
	image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
	image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);

	// copy A to dft_A and pad dft_A with zeros
	cvGetSubRect( dft_A, &tmp, cvRect(0,0, im->width, im->height));
	cvCopy( complexInput, &tmp, NULL );
	if(dft_A->cols > im->width)
	{
		cvGetSubRect( dft_A, &tmp, cvRect(im->width,0, dft_A->cols - im->width, im->height));
		cvZero( &tmp );
	}
	

	// no need to pad bottom part of dft_A with zeros because of
	// use nonzero_rows parameter in cvDFT() call below

	cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );



	// Split Fourier in real and imaginary parts
	cvSplit( dft_A, image_Re, image_Im, 0, 0 );

	// 计算功率谱 Mag = sqrt(Re^2 + Im^2)
	cvPow( image_Re, image_Re, 2.0);
	cvPow( image_Im, image_Im, 2.0);
	cvAdd( image_Re, image_Im, image_Re, NULL);
	cvPow( image_Re, image_Re, 0.5 );

	// 计算 log(1 + Mag)
	cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
	cvLog( image_Re, image_Re ); // log(1 + Mag)


	// 重新安排四个象限，使得原点在图像中心
	cvShiftDFT( image_Re, image_Re );

	// 调整显示象素的区间，保证最大值为白色，最小值为黑色
	cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
	cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));
	

	IplImage* pFFTImage = NULL;
	pFFTImage = cvCreateImage(cvGetSize(image_Re),8,1);
	cvScale(image_Re,pFFTImage,255,0);


	if(im){cvReleaseImage(&im); im = NULL;}
	if(imaginaryInput){cvReleaseImage(&imaginaryInput); imaginaryInput = NULL;}
	if(realInput){cvReleaseImage(&realInput); realInput = NULL;}
	if(complexInput){cvReleaseImage(&complexInput); complexInput = NULL;}

	if(dft_A){cvReleaseMat(&dft_A); dft_A = NULL;}
	if(image_Re){cvReleaseImage(&image_Re); image_Re = NULL;}
	if(image_Im){cvReleaseImage(&image_Im); image_Im = NULL;}
	return pFFTImage;

}



/*
src IPL_DEPTH_64F
dst IPL_DEPTH_8U
*/
void fft2shift(IplImage *src, IplImage *dst)
{
	IplImage *image_Re = 0, *image_Im = 0;
	int nRow, nCol, i, j, cy, cx;
	double scale, shift, tmp13, tmp24;
	image_Re = cvCreateImage(cvGetSize(src), IPL_DEPTH_64F, 1);
	//Imaginary part
	image_Im = cvCreateImage(cvGetSize(src), IPL_DEPTH_64F, 1);
	cvSplit( src, image_Re, image_Im, 0, 0 );

	// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
	cvPow( image_Re, image_Re, 2.0);
	cvPow( image_Im, image_Im, 2.0);
	cvAdd( image_Re, image_Im, image_Re);
	cvPow( image_Re, image_Re, 0.5 );

	// Compute log(1 + Mag)
	cvAddS( image_Re, cvScalar(1.0), image_Re ); // 1 + Mag
	cvLog( image_Re, image_Re ); // log(1 + Mag)

	// Rearrange the quadrants of Fourier image so that the origin is at
	// the image center
	nRow = src->height;
	nCol = src->width;
	cy = nRow/2; // image center
	cx = nCol/2;
	for( j = 0; j < cy; j++ ){
		for( i = 0; i < cx; i++ ){
			tmp13 = CV_IMAGE_ELEM( image_Re, double, j, i);
			CV_IMAGE_ELEM( image_Re, double, j, i) = CV_IMAGE_ELEM(
				image_Re, double, j+cy, i+cx);
			CV_IMAGE_ELEM( image_Re, double, j+cy, i+cx) = tmp13;

			tmp24 = CV_IMAGE_ELEM( image_Re, double, j, i+cx);
			CV_IMAGE_ELEM( image_Re, double, j, i+cx) =
				CV_IMAGE_ELEM( image_Re, double, j+cy, i);
			CV_IMAGE_ELEM( image_Re, double, j+cy, i) = tmp24;
		}
	}

	double minVal = 0, maxVal = 0;
	// Localize minimum and maximum values
	cvMinMaxLoc( image_Re, &minVal, &maxVal );

	//printf("\nmin = %g, max = %g\n", minVal, maxVal);

	//printf("\n Image size : %i x %i pixels\n", image_Re->width, image_Re->height);

	// Normalize image (0 - 255) to be observed as an u8 image
	scale = 255/(maxVal - minVal);
	shift = -minVal * scale;
	cvConvertScale( image_Re, dst, scale, shift);
	cvReleaseImage(&image_Re);
	cvReleaseImage(&image_Im);
}


/*
src IPL_DEPTH_8U
dst IPL_DEPTH_64F
*/
int fft2(IplImage* pSrcImg, IplImage* pFFTImg)
{
	IplImage *image_Re = 0, *image_Im = 0, *Fourier = 0;
	// int i, j;
	// double Re, Im;
	image_Re = cvCreateImage(cvGetSize(pSrcImg), IPL_DEPTH_64F, 1);
	//Imaginary part
	image_Im = cvCreateImage(cvGetSize(pSrcImg), IPL_DEPTH_64F, 1);
	//2 channels (image_Re, image_Im)
	Fourier = cvCreateImage(cvGetSize(pSrcImg), IPL_DEPTH_64F, 2);

	// Real part conversion from u8 to 64f (double)
	cvConvertScale(pSrcImg, image_Re, 1, 0);

	// Imaginary part (zeros)
	cvZero(image_Im);

	// Join real and imaginary parts and stock them in Fourier image
	cvMerge(image_Re, image_Im, 0, 0, Fourier);

	// Application of the forward Fourier transform
	cvDFT(Fourier, pFFTImg, CV_DXT_FORWARD);

	cvReleaseImage(&image_Re);
	cvReleaseImage(&image_Im);
	cvReleaseImage(&Fourier);



	return 0;
}

//fft transform
int ImageQuality_FFT2(IplImage* pSrcImg,IplImage* pDstImg)
{
	IplImage* pGrayImg = NULL;IplImage* Fourier = NULL;
	pGrayImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	if(pSrcImg->nChannels == 3) cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);
	else cvCopy(pSrcImg,pGrayImg);

//	cvSmooth(pGrayImg,pGrayImg,CV_GAUSSIAN);
	Fourier = cvCreateImage(cvGetSize(pSrcImg),IPL_DEPTH_64F,2);
	fft2(pGrayImg,Fourier);
	fft2shift(Fourier,pDstImg);

	if(Fourier){cvReleaseImage(&Fourier);Fourier = NULL;}
	if(pGrayImg){cvReleaseImage(&pGrayImg); pGrayImg = NULL;}
	return 0;
}

//实现傅里叶负变换
int ImageQuality_IFFT(IplImage* pSrcImg,IplImage* pDstImg)
{
	IplImage* pGrayImg = NULL;
	IplImage* Fourier = NULL;IplImage* pTmpImg = NULL;
	IplImage* ImageIm = NULL;IplImage* ImageRe = NULL;
	Fourier = cvCreateImage(cvGetSize(pSrcImg),IPL_DEPTH_64F,2);
	pTmpImg = cvCreateImage(cvGetSize(pSrcImg),IPL_DEPTH_64F,2);
	ImageRe = cvCreateImage(cvGetSize(pSrcImg),IPL_DEPTH_64F,1);
	ImageIm = cvCreateImage(cvGetSize(pSrcImg),IPL_DEPTH_64F,1);
	pGrayImg = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	if(pSrcImg->nChannels == 3) cvCvtColor(pSrcImg,pGrayImg,CV_BGR2GRAY);
	else cvCopy(pSrcImg,pGrayImg);

	double m,M;
	double scale;
	double shift;

	fft2(pGrayImg,Fourier);
	cvDFT(Fourier,pTmpImg,CV_DXT_INV_SCALE);//实现傅里叶逆变换
	cvSplit(pTmpImg,ImageRe,ImageIm,0,0);
	cvPow(ImageRe,ImageRe,2);
	cvPow(ImageIm,ImageIm,2);
	cvAdd(ImageRe,ImageIm,ImageRe,NULL);
	cvPow(ImageRe,ImageRe,0.5);
	cvMinMaxLoc(ImageRe,&m,&M,NULL,NULL);
	scale = 255/(M - m);
	shift = -m * scale;
	cvConvertScale(ImageRe,pDstImg,scale,shift);

	//cvNamedWindow("orc",1);
	//cvShowImage("orc",pGrayImg);
	//cvNamedWindow("fft",1);
	//cvShowImage("fft",pDstImg);

	//cvSub(pGrayImg,pDstImg,pGrayImg);
	//cvNamedWindow("111",1);
	//cvShowImage("111",pGrayImg);
	//cvWaitKey(-1);
	//cvDestroyAllWindows();

	if(Fourier){cvReleaseImage(&Fourier);Fourier = NULL;}
	if(pTmpImg){cvReleaseImage(&pTmpImg);pTmpImg = NULL;}
	if(ImageIm){cvReleaseImage(&ImageIm);ImageIm = NULL;}
	if(ImageRe){cvReleaseImage(&ImageRe);ImageRe = NULL;}
	if(pGrayImg){cvReleaseImage(&pGrayImg); pGrayImg = NULL;}
	return 0;

}

void quickAdaptiveThreshold(unsigned char* grayscale, unsigned char*& thres, int width, int height )   
{   
	
/**           /  
*            | FOREGROUND, if pn < ((gs(n) + gs(n-w)) / (2*s)) *  
* color(n) = |                     ((100-t)/100)  
*            | BACKGROUND_QR, otherwise  
*            \  
* where pn = gray value of current pixel,  
*        s = width of moving average, and  
*        t = threshold percentage of brightness range  
*    gs(n) = gs(n-1) * (1-1/s) + pn  
*    gs(n-w) = gs-value of pixel above current pixel  
*  
    */  
    int t = 15;    
    int s = width >> 3; // s: number of pixels in the moving average (w = image width)   
    const int S = 9; // integer shift, needed to avoid floating point operations   
    const int power2S = 1 << S;   
    // for speedup: multiply all values by 2^s, and use integers instead of floats   
    int factor = power2S * (100-t) / (100*s); // multiplicand for threshold   
    int gn = 127 * s; // initial value of the moving average (127 = average gray value)   
    int q = power2S - power2S / s; // constant needed for average computation   
    int pn, hn;   
    unsigned char *scanline = NULL;   
    int *prev_gn = NULL;   
	int x,y;
    prev_gn = new int[width];   
    for (int i = 0; i < width; i++) {   
        prev_gn[i] = gn;   
    }   
 //   thres = new unsigned char[width*height];   
    for (y = 0; y < height; y ++ )   
    {   
        int yh = y * width;   
        scanline = grayscale + y * width;   
        for ( x = 0; x <width; x ++ )   
        {   
            pn = scanline[x] ;   
            gn = ((gn * q) >> S) + pn;    
            hn = (gn + prev_gn[x]) >> 1;   
            prev_gn[x] = gn;           
            pn < (hn*factor) >> S ? thres[yh+x] = 0 : thres[yh+x] = 0xff;   
        }   
        y ++ ;   
        if ( y == height)   
            break;   
        yh = y * width;   
        scanline = grayscale + y * width;   
        for ( x = width-1; x >= 0; x --)   
        {   
            pn = scanline[x] ;   
            gn = ((gn * q) >> S) + pn;    
            hn = (gn + prev_gn[x]) >> 1;   
            prev_gn[x] = gn;           
            pn < (hn*factor) >> S ? thres[yh+x] = 0 : thres[yh+x] = 0xff;   
        }   
    }   
    delete prev_gn;   
}  

#if 0

#define PERFORMANCE_OPTI
#ifdef PERFORMANCE_OPTI
#ifndef MABS

//fast abs of integer number

#define MABS(x)                (((x)+((x)>>31))^((x)>>31))     

#endif //MABS

int getSobel(IplImage* pSrcImg,IplImage* pSobelImg)

{

	short t0, t1, t2, t3;

	int grad, gradx, grady;

	unsigned char *pline = NULL;

	unsigned char *pTempY_Line0, *pTempY_Line1, *pTempY_Line2;

	int x, y;

	int width  = pSrcImg->width;

	int height = pSrcImg->height;

	int nPitch = pSrcImg->widthStep;

	unsigned char *pTempY = NULL, *pSrcY = (unsigned char *)pSrcImg->imageData;	

	int res = 1;	



	// 中间行

	pTempY = pSrcY + (height-2)*nPitch + (width-1);

	pline = (unsigned char *)pSobelImg->imageData + width * (height-1) - 1;

	for (y = height-2; y >= 1; --y, pTempY-=nPitch)

	{

		pTempY_Line1 = pTempY;

		pTempY_Line0 = pTempY_Line1 - nPitch;

		pTempY_Line2 = pTempY_Line1 + nPitch;



		// 行尾

		t0 = pTempY_Line0[-1];

		t3 = pTempY_Line2[0];

		t2 = pTempY_Line2[-1];

		t1 = pTempY_Line0[0];

		t3 -= t0;

		t0 = pTempY_Line0[0];

		t2 -= t1;

		t1 = pTempY_Line2[0];

		gradx = t3 - t2;

		grady = t3 + t2;

		t2 = pTempY_Line1[-1];

		t3 = pTempY_Line1[0];

		t1 -= t0;

		grady += t1*2;

		t3 -= t2;

		gradx += t3*2;

		grad = (MABS(gradx) + MABS(grady)) >> 1;

		*(pline--) = (unsigned char)(grad>254 ? 254:grad);

		--pTempY_Line1;

		--pTempY_Line0;

		--pTempY_Line2;



		// 行中

		for (x = width-2; x >= 1; --x)

		{

			t0 = pTempY_Line0[-1];

			t3 = pTempY_Line2[1];

			t2 = pTempY_Line2[-1];

			t1 = pTempY_Line0[1];

			t3 -= t0;

			t0 = pTempY_Line0[0];

			t2 -= t1;

			t1 = pTempY_Line2[0];

			gradx = t3 - t2;

			grady = t3 + t2;

			t2 = pTempY_Line1[-1];

			t3 = pTempY_Line1[1];

			t1 -= t0;

			grady += t1*2;

			t3 -= t2;

			gradx += t3*2;

			grad = (MABS(gradx) + MABS(grady)) >> 1;

			*(pline--) = (unsigned char)(grad>254 ? 254:grad);

			--pTempY_Line1;

			--pTempY_Line0;

			--pTempY_Line2;

		}



		// 行首

		t0 = pTempY_Line0[0];

		t3 = pTempY_Line2[1];

		t2 = pTempY_Line2[0];

		t1 = pTempY_Line0[1];

		t3 -= t0;

		t0 = pTempY_Line0[0];

		t2 -= t1;

		t1 = pTempY_Line2[0];

		gradx = t3 - t2;

		grady = t3 + t2;

		t2 = pTempY_Line1[0];

		t3 = pTempY_Line1[1];

		t1 -= t0;

		grady += t1*2;

		t3 -= t2;

		gradx += t3*2;

		grad = (MABS(gradx) + MABS(grady)) >> 1;

		*(pline--) = (unsigned char)(grad>254 ? 254:grad);

	}



	// 最后一行

	pTempY_Line1 = pSrcY + (height-1)*nPitch + (width-1);

	pTempY_Line0 = pTempY_Line1 - nPitch;

	pTempY_Line2 = pTempY_Line1 + nPitch;

	pline = (unsigned char *)pSobelImg->imageData + width * (height-1) + (width - 1);

	// 行尾

	t0 = pTempY_Line0[-1];

	t3 = pTempY_Line1[0];

	t2 = pTempY_Line1[-1];

	t1 = pTempY_Line0[0];

	t3 -= t0;

	t0 = pTempY_Line0[0];

	t2 -= t1;

	t1 = pTempY_Line1[0];

	gradx = t3 - t2;

	grady = t3 + t2;

	t2 = pTempY_Line1[-1];

	t3 = pTempY_Line1[0];

	t1 -= t0;

	grady += t1*2;

	t3 -= t2;

	gradx += t3*2;

	grad = (MABS(gradx) + MABS(grady)) >> 1;

	*(pline--) = (unsigned char)(grad>254 ? 254:grad);

	--pTempY_Line1;

	--pTempY_Line0;

	--pTempY_Line2;



	// 行中

	for (x = width-2; x >= 1; --x)

	{

		t0 = pTempY_Line0[-1];

		t3 = pTempY_Line1[1];

		t2 = pTempY_Line1[-1];

		t1 = pTempY_Line0[1];

		t3 -= t0;

		t0 = pTempY_Line0[0];

		t2 -= t1;

		t1 = pTempY_Line1[0];

		gradx = t3 - t2;

		grady = t3 + t2;

		t2 = pTempY_Line1[-1];

		t3 = pTempY_Line1[1];

		t1 -= t0;

		grady += t1*2;

		t3 -= t2;

		gradx += t3*2;

		grad = (MABS(gradx) + MABS(grady)) >> 1;

		*(pline--) = (unsigned char)(grad>254 ? 254:grad);

		--pTempY_Line1;

		--pTempY_Line0;

		--pTempY_Line2;

	}



	// 行首

	t0 = pTempY_Line1[0];

	t3 = pTempY_Line2[1];

	t2 = pTempY_Line2[0];

	t1 = pTempY_Line1[1];

	t3 -= t0;

	t0 = pTempY_Line1[0];

	t2 -= t1;

	t1 = pTempY_Line2[0];

	gradx = t3 - t2;

	grady = t3 + t2;

	t2 = pTempY_Line1[0];

	t3 = pTempY_Line1[1];

	t1 -= t0;

	grady += t1*2;

	t3 -= t2;

	gradx += t3*2;

	grad = (MABS(gradx) + MABS(grady)) >> 1;

	*(pline--) = (unsigned char)(grad>254 ? 254:grad);





	// 第一行

	pTempY_Line1 = pSrcY + (width-1);

	pTempY_Line0 = pTempY_Line1 - nPitch;

	pTempY_Line2 = pTempY_Line1 + nPitch;

	pline = (unsigned char *)pSobelImg->imageData + (width-1);

	// 行尾

	t0 = pTempY_Line0[-1];

	t3 = pTempY_Line1[0];

	t2 = pTempY_Line1[-1];

	t1 = pTempY_Line0[0];

	t3 -= t0;

	t0 = pTempY_Line0[0];

	t2 -= t1;

	t1 = pTempY_Line1[0];

	gradx = t3 - t2;

	grady = t3 + t2;

	t2 = pTempY_Line1[-1];

	t3 = pTempY_Line1[0];

	t1 -= t0;

	grady += t1*2;

	t3 -= t2;

	gradx += t3*2;

	grad = (MABS(gradx) + MABS(grady)) >> 1;

	*(pline--) = (unsigned char)(grad>254 ? 254:grad);

	--pTempY_Line1;

	--pTempY_Line0;

	--pTempY_Line2;



	// 行中

	for (x = width-2; x >= 1; --x)

	{

		t0 = pTempY_Line1[-1];

		t3 = pTempY_Line2[1];

		t2 = pTempY_Line2[-1];

		t1 = pTempY_Line1[1];

		t3 -= t0;

		t0 = pTempY_Line1[0];

		t2 -= t1;

		t1 = pTempY_Line2[0];

		gradx = t3 - t2;

		grady = t3 + t2;

		t2 = pTempY_Line1[-1];

		t3 = pTempY_Line1[1];

		t1 -= t0;

		grady += t1*2;

		t3 -= t2;

		gradx += t3*2;

		grad = (MABS(gradx) + MABS(grady)) >> 1;

		*(pline--) = (unsigned char)(grad>254 ? 254:grad);

		--pTempY_Line1;

		--pTempY_Line0;

		--pTempY_Line2;

	}



	// 行首

	t0 = pTempY_Line1[0];

	t3 = pTempY_Line2[1];

	t2 = pTempY_Line2[0];

	t1 = pTempY_Line1[1];

	t3 -= t0;

	t0 = pTempY_Line1[0];

	t2 -= t1;

	t1 = pTempY_Line2[0];

	gradx = t3 - t2;

	grady = t3 + t2;

	t2 = pTempY_Line1[0];

	t3 = pTempY_Line1[1];

	t1 -= t0;

	grady += t1*2;

	t3 -= t2;

	gradx += t3*2;

	grad = (MABS(gradx) + MABS(grady)) >> 1;

	*(pline--) = (unsigned char)(grad>254 ? 254:grad);



	return res;

}

#endif //PERFORMANCE_OPTI

#else

//原始版本边框会有白线,需要先做初始化
int getSobel(IplImage* pSrcImg,IplImage* pSobelImg)
{
	if (pSrcImg->nChannels != 1)  //只做灰度图
	{
		printf("WARN: in func getSobelEdge : pSrcImg->nChannels is not 1 \n");
		return -1;
	}
	if (pSobelImg==NULL)
	{
		printf("WARN: in func getSobelEdge : pSobelImg is NULL \n");
		return -1;
	}

	int		CoefArrayh[9]={-1,0,1,
		-2,0,2,
		-1,0,1 };//模板数组
	int		CoefArrayv[9]={-1,-2,-1,
		0, 0, 0,
		1, 2, 1};

	int	x,y,i,j;
	unsigned int TempSrcDatah[9],TempSrcDatav[9];
	int	TempNumh=0,TempNumv=0;
	int lHeight=0, lWidth=0, lLineBytes=0;
	int H_sOne=0, W_sOne=0;
	int K1=0, K2=0;
	uchar* TempOne = NULL, *TempTwo=NULL;

	unsigned char *pBitmap = (unsigned char *)pSrcImg->imageData;
	unsigned char *pSobel = (unsigned char *)pSobelImg->imageData;

	lHeight = pSrcImg->height;
	lWidth = pSrcImg->width;
	lLineBytes = pSrcImg->widthStep;
	H_sOne = lHeight-1;
	W_sOne = lWidth-1;
	uchar* pSrcOffSet = pBitmap;
	uchar* pSobelOffSet = pSobel;

	pSrcOffSet+=lLineBytes;
	pSobelOffSet+=lLineBytes;

	TempOne = pBitmap;
	TempTwo = pSobel;

	//first corner
	K1 = K2 = 3;
	TempSrcDatah[5] = (*(TempOne+1))<<1;
	TempSrcDatah[8] = (*(TempOne+1+lLineBytes));
	TempSrcDatav[7] = (*(TempOne+lLineBytes))<<1;
	TempSrcDatav[8] = TempSrcDatah[8];

	TempNumh = TempSrcDatah[5] + TempSrcDatah[8];
	TempNumv = TempSrcDatav[7] + TempSrcDatav[8];

	TempNumh /= 3;
	TempNumv /= 3;

	TempNumv = sqrt((float)((TempNumh*TempNumh<<1)+(TempNumv*TempNumv<<1)));


	if (TempNumv > 255.0)
	{
		*TempTwo = (uchar)255;
	}
	//else if (TempNumv < 0.0) 
	//	*pSobelData = (unsigned char)abs(TempNumv);
	else 
		*TempTwo = (uchar)TempNumv;

	//second corner
	TempOne = pBitmap + lWidth - 1;
	TempTwo = pSobel + lWidth - 1;
	K1 = -3; K2 = 3;

	TempSrcDatah[3] = (*(TempOne-1))<<1;
	TempSrcDatah[3] = -TempSrcDatah[3];

	TempSrcDatah[6] = (*(TempOne-1+lLineBytes));
	TempSrcDatah[6] = -TempSrcDatah[6];

	TempSrcDatav[6] = -TempSrcDatah[6];
	TempSrcDatav[7] = (*(TempOne+lLineBytes))<<1;

	TempNumh = TempSrcDatah[3] + TempSrcDatah[6];
	TempNumv = TempSrcDatav[6] + TempSrcDatav[7];

	TempNumh /= -3;
	TempNumv /= 3;

	TempNumv = sqrt((float)((TempNumh*TempNumh<<1)+(TempNumv*TempNumv<<1)));


	if (TempNumv > 255.0)
	{
		*TempTwo = (uchar)255;
	}
	//else if (TempNumv < 0.0) 
	//	*pSobelData = (unsigned char)abs(TempNumv);
	else 
		*TempTwo = (uchar)TempNumv;

	//third corner
	TempOne = pBitmap + (lHeight-1)*lLineBytes;
	TempTwo = pSobel + (lHeight-1)*lLineBytes;
	K1 = 3; K2 = -3;

	TempSrcDatah[2] = *(TempOne+1-lLineBytes);
	TempSrcDatav[1] = (*(TempOne-lLineBytes))<<1;
	TempSrcDatav[1] = -TempSrcDatav[1];
	TempSrcDatav[2] = -TempSrcDatah[2];
	TempSrcDatah[5] = (*(TempOne+1))<<1;

	TempNumh = TempSrcDatah[2] + TempSrcDatah[5];
	TempNumv = TempSrcDatav[1] + TempSrcDatav[2];

	TempNumh /= 3;
	TempNumv /= -3;

	TempNumv = sqrt((float)((TempNumh*TempNumh<<1)+(TempNumv*TempNumv<<1)));


	if (TempNumv > 255.0)
	{
		*TempTwo = (uchar)255;
	}
	//else if (TempNumv < 0.0) 
	//	*pSobelData = (unsigned char)abs(TempNumv);
	else 
		*TempTwo = (uchar)TempNumv;

	//forth corner
	TempOne = pBitmap + (lHeight-1)*lLineBytes + lWidth - 1;
	TempTwo = pSobel + (lHeight-1)*lLineBytes + lWidth - 1;
	K1 = -3; K2 = -3;

	TempSrcDatah[0] = *(TempOne-1-lLineBytes);
	TempSrcDatah[0] = -TempSrcDatah[0];
	TempSrcDatav[0] = TempSrcDatah[0];


	TempSrcDatav[1] = (*(TempOne-lLineBytes))<<1;
	TempSrcDatav[1] = -TempSrcDatav[1];

	TempSrcDatah[3] = (*(TempOne-1))<<1;
	TempSrcDatah[3] = -TempSrcDatah[3];

	TempNumh = TempSrcDatah[0] + TempSrcDatah[3];
	TempNumv = TempSrcDatav[0] + TempSrcDatav[1];

	TempNumh /= -3;
	TempNumv /= -3;

	TempNumv = sqrt((float)((TempNumh*TempNumh<<1)+(TempNumv*TempNumv<<1)));


	if (TempNumv > 255.0)
	{
		*TempTwo = (uchar)255;
	}
	//else if (TempNumv < 0.0) 
	//	*pSobelData = (unsigned char)abs(TempNumv);
	else 
		*TempTwo = (uchar)TempNumv;



	//first row
	for (x=1; x<W_sOne; x++){
		TempNumh = 0;
		TempNumv = 0;
		K1 = 0;
		K2 = 4;
		for (i=-1; i<=1; i++){
			for (j=-1; j<=1; j++){
				unsigned int TempVal = 0;
				if (i<0 || i>H_sOne || x+j<0 || x+j>W_sOne)
					continue;
				TempVal = pBitmap[i*lLineBytes+x+j];
				TempNumh += TempVal*CoefArrayh[(i+1)*3+j+1];
				TempNumv += TempVal*CoefArrayv[(i+1)*3+j+1];
			}
		}

		TempNumv = TempNumv>>2;
		TempNumv = sqrt((float)((TempNumh*TempNumh<<1)+(TempNumv*TempNumv<<1))); //use edge mount
		if (TempNumv > 255.0)
		{
			pSobel[x] = (uchar)255;
		}
		//else if (TempNumv < 0.0) 
		//	pSobel[y*lLineBytes+x] = (unsigned char)abs(TempNumv);
		else 
			pSobel[x] = (uchar)TempNumv;
	}


	//final row
	for (x=1; x<W_sOne; x++)
	{
		TempNumh = 0;
		TempNumv = 0;
		K1 = 0;
		K2 = 4;
		y = H_sOne;
		for (i=-1; i<=1; i++)
		{
			for (j=-1; j<=1; j++)
			{
				unsigned int TempVal = 0;
				if (y+i<0 || y+i>H_sOne || x+j<0 || x+j>W_sOne)
					continue;
				TempVal = pBitmap[(y+i)*lLineBytes+x+j];
				TempNumh += TempVal*CoefArrayh[(i+1)*3+j+1];
				TempNumv += TempVal*CoefArrayv[(i+1)*3+j+1];
			}
		}

		TempNumv = TempNumv>>2;
		TempNumv = sqrt((float)((TempNumh*TempNumh<<1)+(TempNumv*TempNumv<<1))); //use edge mount
		if (TempNumv > 255.0)
		{
			pSobel[y*lLineBytes+x] = (uchar)255;
		}
		//else if (TempNumv < 0.0) 
		//	pSobel[y*lLineBytes+x] = (unsigned char)abs(TempNumv);
		else 
			pSobel[y*lLineBytes+x] = (uchar)TempNumv;
	}

	//left column
	for (y=1; y<H_sOne; y++){
		TempNumh = 0;
		TempNumv = 0;
		K1 = -4;
		K2 = 0;
		for (i=-1; i<=1; i++){
			for (j=-1; j<=1; j++){
				if (y+i<0 || y+i>H_sOne || j<0 || j>W_sOne)
					continue;
				TempNumh += pBitmap[(y+i)*lLineBytes+j]*CoefArrayh[(i+1)*3+j+1];
				TempNumv += pBitmap[(y+i)*lLineBytes+j]*CoefArrayv[(i+1)*3+j+1];
			}
		}
		TempNumh = -(TempNumh>>2);
		TempNumv = sqrt((float)((TempNumh*TempNumh<<1)+(TempNumv*TempNumv<<1))); //use edge mount
		if (TempNumv > 255.0)
		{
			pSobel[y*lLineBytes] = (uchar)255;
		}
		//else if (TempNumv < 0.0) 
		//	pSobel[y*lLineBytes] = (uchar)ABS((int)TempNumv);
		else 
			pSobel[y*lLineBytes] = (uchar)TempNumv;

	}
	//right column

	for (y=1; y<H_sOne; y++)
	{
		x=W_sOne;
		TempNumh = 0;
		TempNumv = 0;
		K1 = -4;
		K2 = 0;
		for (i=-1; i<=1; i++)
		{
			for (j=-1; j<=1; j++)
			{
				if (y+i<0 || y+i>H_sOne || x+j<0 || x+j>W_sOne)
					continue;
				TempNumh += pBitmap[(y+i)*lLineBytes+x+j]*CoefArrayh[(i+1)*3+j+1];
				TempNumv += pBitmap[(y+i)*lLineBytes+x+j]*CoefArrayv[(i+1)*3+j+1];
			}
		}
		TempNumh = -(TempNumh>>2);
		TempNumv = sqrt((float)((TempNumh*TempNumh<<1)+(TempNumv*TempNumv<<1))); //use edge mount
		if (TempNumv > 255.0)
		{
			pSobel[y*lLineBytes+x] = (uchar)255;
		}
		//else if (TempNumv < 0.0) 
		//	pSobel[y*lLineBytes+x] = (uchar)abs(TempNumv);
		else 
			pSobel[y*lLineBytes+x] = (uchar)TempNumv;

	}




	//Major

	for (y=1; y<H_sOne; y++)
	{
		uchar* pSrcData = pSrcOffSet;
		uchar* pSobelData = pSobelOffSet;
		pSrcData++;
		pSobelData++;
		for (x=1; x<W_sOne; x++)
		{


			TempNumh = 0;
			TempNumv = 0;
			TempOne = pSrcData-lLineBytes-1;
			TempSrcDatah[0] = *(TempOne);
			TempSrcDatah[0] = -TempSrcDatah[0];

			TempSrcDatav[1] = -(*(TempOne+1)<<1);
			TempSrcDatah[2] = *(TempOne+2);
			TempSrcDatav[2] = -TempSrcDatah[2];

			TempSrcDatah[3] = *(TempOne+lLineBytes);
			TempSrcDatah[3] = -(TempSrcDatah[3]<<1);

			TempSrcDatah[5] = (*(pSrcData+1))<<1;
			TempOne = pSrcData+lLineBytes;

			TempSrcDatav[6] = *(TempOne-1);
			TempSrcDatah[6] = TempSrcDatav[6]*(-1);

			TempSrcDatav[7] = *(TempOne)<<1;
			TempSrcDatah[8] = *(TempOne+1);

			TempNumh = TempSrcDatah[0] + TempSrcDatah[2] + TempSrcDatah[3] + TempSrcDatah[5] + TempSrcDatah[6] + TempSrcDatah[8];
			TempNumv = TempSrcDatah[0] + TempSrcDatav[1] + TempSrcDatav[2] + TempSrcDatav[6] + TempSrcDatav[7] + TempSrcDatah[8];
			TempNumv = sqrt((float)((TempNumh*TempNumh<<1)+(TempNumv*TempNumv<<1))); //use edge mount
			if (TempNumv > 255.0)
			{
				*pSobelData = (uchar)255;
			}
			//else if (TempNumv < 0.0) 
			//	*pSobelData = (unsigned char)abs(TempNumv);
			else 
				*pSobelData = (uchar)TempNumv;

			pSrcData++;
			pSobelData++;
		}
		pSrcOffSet+=lLineBytes;
		pSobelOffSet+=lLineBytes;
	}
#if 1
	i = lHeight - 1;
	for (j=0; j<lWidth; j++)
	{
		pSobel[j] = 0;
		pSobel[i*lLineBytes+j] = 0;
	}
	j = lWidth - 1;
	for (i=0; i<lHeight; i++)
	{
		pSobel[i*lLineBytes] = 0;
		pSobel[i*lLineBytes+j] = 0;
	}
#endif

	return 1;
}

#endif





double efficientLocalMean(const int x,const int y,const int k,IplImage* pSrcImg) 
{
	int k2=k/2;

	int nWid = pSrcImg->width;
	int nHei = pSrcImg->height;
	//wanting average over area: (y-k2,x-k2) ... (y+k2-1, x+k2-1)
	int starty=y-k2;
	int startx=x-k2;
	int stopy=y+k2-1;
	int stopx=x+k2-1;

	if(starty<0) starty=0;
	if(startx<0) startx=0;
	if(stopx>nWid-1) stopx=nWid-1;
	if(stopy>nHei-1) stopy=nHei-1;

	double unten, links, oben, obenlinks;

	if(startx-1<0) links=0; 
	else links = CV_IMAGE_ELEM(pSrcImg,uchar,stopy,startx-1);
//	else links=laufendeSumme(startx-1,stopy,0);

	if(starty-1<0) oben=0;
	else oben = CV_IMAGE_ELEM(pSrcImg,uchar,starty-1,startx);
//	else oben=laufendeSumme(stopx,starty-1,0);

	if((starty-1 < 0) || (startx-1 <0)) obenlinks=0;
	else obenlinks = CV_IMAGE_ELEM(pSrcImg,uchar,starty-1,startx-1);
//	else obenlinks=laufendeSumme(startx-1,starty-1,0);

	unten = CV_IMAGE_ELEM(pSrcImg,uchar,stopy,stopx-1);
//	unten=laufendeSumme(stopx,stopy,0);

	//   cout << "obenlinks=" << obenlinks << " oben=" << oben << " links=" << links << " unten=" <<unten << endl;
	int counter=(stopy-starty+1)*(stopx-startx+1);
	return (unten-links-oben+obenlinks)/counter;
}



int coarseness(IplImage* pSrcImg,IplImage* pData) 
{
	if(pSrcImg->nChannels != 1) return -1;
	if(!pData )return -1;
	//	if(pDstImg->nChannels != 1) return -1;
	//init
	int nWid = pSrcImg->width;
	int nHei = pSrcImg->height;

	//这部分感觉它没写完整
	//// initialize for running sum calculation
	//double links, oben, obenlinks;

	//for(int y=0;y<nHei;++y) {
	//	for(int x=0;x<nWid;++x) {
	//		if(x<1) links=0;
	//		else links=laufendeSumme(x-1,y,0);

	//		if(y<1) oben=0;
	//		else oben=laufendeSumme(x,y-1,0);

	//		if(y<1 || x<1) obenlinks=0;
	//		else obenlinks=laufendeSumme(x-1,y-1,0);

	//		laufendeSumme(x,y,0)=image(x,y,z)+links+oben-obenlinks;
	//	}
	//}
	/*ImageFeature Ak(xDim,yDim,5);
	ImageFeature Ekh(xDim,yDim,5);
	ImageFeature Ekv(xDim,yDim,5);

	ImageFeature Sbest(xDim,yDim,1);*/


	double** Ak = NULL;
	double** Ekh = NULL;
	double** Ekv = NULL;

	Ak = new double*[5];
	Ekh = new double*[5];
	Ekv = new double*[5];
	for(int i = 0; i < 5; i ++)
	{
		Ak[i] = new double[nWid*nHei];
		Ekh[i] = new double[nWid*nHei];
		Ekv[i] = new double[nWid*nHei];
	}
	


	//step 1
	int lenOfk=1;
	for(int k=1;k<=5;++k) {
		lenOfk*=2;
		for(int y=0;y<nHei;++y) {
			for(int x=0;x<nWid;++x) {
				Ak[k-1][y*nWid+x]=efficientLocalMean(x,y,lenOfk,pSrcImg);
			}
		}
	}

//	DBG(25) << "  ... step 2 ... " << endl;
	//step 2
	lenOfk=1;
	for(int k=1;k<=5;++k) {
		int k2=lenOfk;
		lenOfk*=2;
		for(int y=0;y<nHei;++y) {
			for(int x=0;x<nWid;++x) {

				int posx1=x+k2;
				int posx2=x-k2;

				int posy1=y+k2;
				int posy2=y-k2;
				if(posx1<nWid && posx2>=0) 
				{
				//	Ekh(x,y,k-1)=fabs(Ak(posx1,y,k-1)-Ak(posx2,y,k-1));
					Ekh[k-1][y*nWid+x] = fabs(Ak[k-1][y*nWid+posx1] - Ak[k-1][y*nWid+posx2]);
				} 
				else 
				{
				//	Ekh(x,y,k-1)=0;
					Ekh[k-1][y*nWid+x] = 0;
				}
				if(posy1<nHei && posy2>=0) 
				{
				//	Ekv(x,y,k-1)=fabs(Ak(x,posy1,k-1)-Ak(x,posy2,k-1));
					Ekv[k-1][y*nWid+x] = fabs(Ak[k-1][posy1*nWid+x] - Ak[k-1][posy2*nWid+x]);
				} 
				else 
				{
				//	Ekv(x,y,k-1)=0;
					Ekv[k-1][y*nWid+x] = 0;
				}
			}
		}
	}
	double sum=0.0;
//	DBG(25) << "  ... step 3 ... " << endl;  
	//step3
	for(int y=0;y<nHei;++y) {
		for(int x=0;x<nWid;++x) {
			double maxE=0;
			int maxk=0;
			for(int k=1;k<=5;++k) 
			{
			//	if(Ekh(x,y,k-1)>maxE) 
				if(Ekh[k-1][y*nWid+x] > maxE)
				{
				//	maxE=Ekh(x,y,k-1);
					maxE = Ekh[k-1][y*nWid+x];
					maxk=k;
				}
			//	if(Ekv(x,y,k-1)>maxE) 
				if(Ekv[k-1][y*nWid+x] > maxE)
				{
				//	maxE=Ekv(x,y,k-1);
					maxE = Ekv[k-1][y*nWid+x];
					maxk=k;
				}
			}
		//	Sbest(x,y,0)=maxk;
			CV_IMAGE_ELEM(pData,double,y,x) = maxk;
			sum+=maxk;
		}
	}

	sum /=((nHei-32)*(nWid-32));

	for(int i =0; i < 5 ;i ++)
	{
		delete []Ak[i];
		Ak[i] = NULL;
		delete []Ekh[i];
		Ekh[i] = NULL;
		delete []Ekv[i];
		Ekv[i] = NULL;
	}

	delete []Ak;
	delete []Ekh;
	delete []Ekv;
	
//	DBG(25) << "Average Coarseness: " << sum << endl;
	return 0;
}



int convolve(const IplImage * pOriData, int matrix[3][3],IplImage* pEdgeData)
{
	int X, Y;
	int I, J;
	int sumX,sumY,SUM;
	int iWidth = pOriData->width;
	int iHeight = pOriData->height;

	if(pOriData->nChannels != 1 || pEdgeData->nChannels != 1)
		return -1;
	uchar pixel = 0;
	//	memset(pEdgeData,0,sizeof(uchar)*iWidth*iHeight);
	cvZero(pEdgeData);

	/* 3x3 Sobel mask.  Ref:  www.cee.hw.ac.uk/hipr/html/sobel.html */
	

	for(Y = 0; Y < iHeight; Y++) 
	{
		for(X = 0; X < iWidth; X++) 
		{
			/* image boundaries */
			if(Y == 0 || Y == iHeight - 1)
				SUM = 0;
			else if(X == 0 || X == iWidth - 1)
				SUM = 0;

			/* Convolution starts here */
			else   
			{
				/*-------X GRADIENT APPROXIMATION------*/
				for(I = -1; I <= 1; I++)  {
					for(J = -1; J <= 1; J++)  {
						//	sumX = sumX + (int)(pOriData[(Y+J) * iWidth + (X+I)] * SOBEL_X[I+1][J+1]);
						SUM = SUM + (int)(((uchar*)(pOriData->imageData + (Y + J) * pOriData->widthStep))[X + I] * matrix[I+1][J+1]);
					}
				}

				if(SUM > 255) SUM = 255; 
				if(SUM < 0) SUM = 0;

			}
			((uchar*)(pEdgeData->imageData + Y * pEdgeData->widthStep))[X] = (uchar)SUM;
			//	pEdgeData[Y * iWidth + X] = SUM;
		}
	}
	return 0;
}


double getLocalContrast(IplImage* pSrcImg, int xpos, int ypos) 
{
	int nWid = pSrcImg->width;
	int nHei = pSrcImg->height;

	int ystart=::std::max(0,ypos-5);
	int xstart=::std::max(0,xpos-5);
	int ystop=::std::min(nHei,ypos+6);
	int xstop=::std::min(nWid,xpos+6);

	int size=(ystop-ystart)*(xstop-xstart);

	double mean=0.0, sigma=0.0, kurtosis=0.0, tmp;

	for(int y=ystart;y<ystop;++y) {
		for(int x=xstart;x<xstop;++x) {
			tmp = CV_IMAGE_ELEM(pSrcImg,uchar,y,x);
		//	tmp=image(x,y,z);
			mean+=tmp;
			sigma+=tmp*tmp;
		}
	}
	mean/=size;
	sigma/=size;
	sigma-=mean*mean;

	for(int y=ystart;y<ystop;++y) {
		for(int x=xstart;x<xstop;++x) {
			tmp = CV_IMAGE_ELEM(pSrcImg,uchar,y,x) - mean;
		//	tmp=image(x,y,z)-mean;
			tmp*=tmp;
			tmp*=tmp;
			kurtosis+=tmp;
		}
	}
	kurtosis/=size;
	//double alpha4=kurtosis/(sigma*sigma);
	//double contrast=sqrt(sigma)/sqrt(sqrt(alpha4));
	//return contrast;

	/// if we don't have this exception, there are numeric problems!
	/// if the region is homogeneous: kurtosis and sigma are numerically very close to zero
	if(kurtosis<numeric_limits<double>::epsilon()) {
		return 0.0;
	}

	return sigma/sqrt(sqrt(kurtosis));
}


int contrast(IplImage* pSrcImg, IplImage* pData) 
{
	if(pSrcImg->nChannels != 1) return -1;
	if(!pData) return -1;

	int nWid = pSrcImg->width;
	int nHei = pSrcImg->height;

	double min=numeric_limits<double>::max();
	double max=0;
	double tmp;

	for(int x=0;x<nWid;++x) {
		for(int y=0;y<nHei;++y) {
			tmp=getLocalContrast(pSrcImg,x,y);

			CV_IMAGE_ELEM(pData,double,y,x) = tmp;
			min=::std::min(min,tmp);
			max=::std::max(max,tmp);
		}
	}

	return 0;
}

//z means channel
int directionality(IplImage* pSrcImg,IplImage* pData) 
{
	
	if(pSrcImg->nChannels != 1) return -1;
	if(!pData) return -1;
	//init
	int nWid = pSrcImg->width;
	int nHei = pSrcImg->height;

	IplImage* pH = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);
	IplImage* pV = cvCreateImage(cvGetSize(pSrcImg),pSrcImg->depth,1);

	//step1
	int SOBEL_X[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
	int SOBEL_Y[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

	convolve(pSrcImg,SOBEL_X,pH);
	convolve(pSrcImg,SOBEL_Y,pV);

	//step2
	for(int y=0;y < nHei;++y) {
		for(int x=0;x<nWid;++x) {
			//deltaG(x,y,0)=fabs(deltaH(x,y,0))+fabs(deltaV(x,y,0));
			//deltaG(x,y,0)*=0.5;

		//	if(deltaH(x,y,0)!=0.0) {
		//		phi(x,y,0)=atan(deltaV(x,y,0)/deltaH(x,y,0))+(CV_PI/2.0+0.001); //+0.001 because otherwise sometimes getting -6.12574e-17

			if(CV_IMAGE_ELEM(pH,uchar,y,x))
			{
				CV_IMAGE_ELEM(pData,double,y,x) =atan( CV_IMAGE_ELEM(pV,uchar,y,x) * 1.0 / CV_IMAGE_ELEM(pH,uchar,y,x)) + (CV_PI / 2.0 + 0.001);
			//	pData[y*nWid+x] = atan( CV_IMAGE_ELEM(pV,uchar,y,x) * 1.0 / CV_IMAGE_ELEM(pH,uchar,y,x)) + (CV_PI / 2.0 + 0.001);
			}
		}
	}

	if(pH){cvReleaseImage(&pH);pH = NULL;}
	if(pV){cvReleaseImage(&pV);pV = NULL;}

	return 0;
}
}