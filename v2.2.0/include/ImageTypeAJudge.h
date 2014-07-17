#pragma once
#include <cv.h>

namespace ImageTypeAJudge_2_2_0
{
	//************************************
	// Method:    InitImageTypeAJudgeClassify
	// FullName:  InitImageTypeAJudgeClassify
	// Access:    public 
	// Returns:   int 0 TOK other TErrorCode.h 
	// Qualifier:
	// Parameter: char * featPath  训练特到的特征文件
	// Parameter: CvSVM & imagetypeajudge_svm 需要申明的分类器类型
	//************************************
	int InitImageTypeAJudgeClassify(char* featPath);

	//************************************
	// Method:    ImageTypeAJudge
	// FullName:  ImageTypeAJudge
	// Access:    public 
	// Returns:   int 0 TOK other TErrorCode.h
	// Qualifier:
	// Parameter: IplImage * pSrcImg:input image
	// Parameter: int &result:返回A图检测结果：1-A图,0-非A图。
	//************************************
	int ImageTypeAJudge(IplImage* pSrcImg,int &result);

	//************************************
	// Method:    ReleaseImageTypeAJudgeClassify
	// FullName:  ReleaseImageTypeAJudgeClassify
	// Access:    public 
	// Returns:   int 0 TOK other TErrorCode.h 
	// Qualifier:
	//************************************
	int ReleaseImageTypeAJudgeClassify();

	void version();
}