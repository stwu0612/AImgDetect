#pragma once

namespace ImageTypeAJudge_2_0_0
{
//分块
#define CLDSUB 4

//特征种类
#define EHDDIM  80//144
#define CLDDIM 12	
#define BLURDIM 5

//特征选择
#define EHD							1
#define CLD                         1
#define BLUR						1

struct Feat
{
#if EHD	//纹理
	float fEHD[EHDDIM];
	float fsobelEHD[EHDDIM];
	float fBinEHD[EHDDIM];
	float fOptEHD[EHDDIM];
#endif

#if CLD	//颜色
	float fCLD[CLDDIM];
#endif

#if BLUR
	float fBlur[BLURDIM];
#endif
};
}