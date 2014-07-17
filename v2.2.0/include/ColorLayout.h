#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

namespace ImageTypeAJudge_2_2_0
{
//Extracting MPEG-7 CLD
void ColorLayoutExtractor(IplImage *Image, float fCLD[]);

//Computing Similarity of 2 CLD
double CLDDist(int CLD1[], int CLD2[]);
}