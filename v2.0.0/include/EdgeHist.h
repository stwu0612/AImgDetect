#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

namespace ImageTypeAJudge_2_0_0
{
//Extracting MPEG-7 EHD
void EdgeHistExtractor(IplImage *Image, float fEHD[],IplImage* pMskImg = NULL);

//Computing Similarity of 2 EHD
double EHDDist(int EHD1[], int EHD2[]);
}
