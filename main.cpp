// opencv_test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "string"
#include "opencv2\opencv.hpp"
using namespace cv;
using namespace std;


//functions: icvprLabelColor, icvprCcaByTwoPass,  icvprGetRandomColor are opensource code from some website, but I forgot where did I get these codes, I'd be appreciated if someone know where it come from and tell me
cv::Scalar icvprGetRandomColor()
{
    uchar r = 255 * (rand()/(1.0 + RAND_MAX));
    uchar g = 255 * (rand()/(1.0 + RAND_MAX));
    uchar b = 255 * (rand()/(1.0 + RAND_MAX));
    return cv::Scalar(b,g,r) ;
}
void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg)
{
    // connected component analysis (4-component)
    // use two-pass algorithm
    // 1. first pass: label each foreground pixel with a label
    // 2. second pass: visit each labeled pixel and merge neighbor labels
    //
    // foreground pixel: _binImg(x,y) = 1
    // background pixel: _binImg(x,y) = 0


    if (_binImg.empty() ||
            _binImg.type() != CV_8UC1)
    {
        return ;
    }

    // 1. first pass

    _lableImg.release() ;
    _binImg.convertTo(_lableImg, CV_32SC1) ;

    int label = 1 ;  // start by 2
    std::vector<int> labelSet ;
    labelSet.push_back(0) ;   // background: 0
    labelSet.push_back(1) ;   // foreground: 1

    int rows = _binImg.rows - 1 ;
    int cols = _binImg.cols - 1 ;
    for (int i = 1; i < rows; i++)
    {
        int* data_preRow = _lableImg.ptr<int>(i-1) ;
        int* data_curRow = _lableImg.ptr<int>(i) ;
        for (int j = 1; j < cols; j++)
        {
            if (data_curRow[j] == 1)
            {
                std::vector<int> neighborLabels ;
                neighborLabels.reserve(2) ;
                int leftPixel = data_curRow[j-1] ;
                int upPixel = data_preRow[j] ;
                if ( leftPixel > 1)
                {
                    neighborLabels.push_back(leftPixel) ;
                }
                if (upPixel > 1)
                {
                    neighborLabels.push_back(upPixel) ;
                }

                if (neighborLabels.empty())
                {
                    labelSet.push_back(++label) ;  // assign to a new label
                    data_curRow[j] = label ;
                    labelSet[label] = label ;
                }
                else
                {
                    std::sort(neighborLabels.begin(), neighborLabels.end()) ;
                    int smallestLabel = neighborLabels[0] ;
                    data_curRow[j] = smallestLabel ;

                    // save equivalence
                    for (size_t k = 1; k < neighborLabels.size(); k++)
                    {
                        int tempLabel = neighborLabels[k] ;
                        int& oldSmallestLabel = labelSet[tempLabel] ;
                        if (oldSmallestLabel > smallestLabel)
                        {
                            labelSet[oldSmallestLabel] = smallestLabel ;
                            oldSmallestLabel = smallestLabel ;
                        }
                        else if (oldSmallestLabel < smallestLabel)
                        {
                            labelSet[smallestLabel] = oldSmallestLabel ;
                        }
                    }
                }
            }
        }
    }

    // update equivalent labels
    // assigned with the smallest label in each equivalent label set
    for (size_t i = 2; i < labelSet.size(); i++)
    {
        int curLabel = labelSet[i] ;
        int preLabel = labelSet[curLabel] ;
        while (preLabel != curLabel)
        {
            curLabel = preLabel ;
            preLabel = labelSet[preLabel] ;
        }
        labelSet[i] = curLabel ;
    }


    // 2. second pass
    for (int i = 0; i < rows; i++)
    {
        int* data = _lableImg.ptr<int>(i) ;
        for (int j = 0; j < cols; j++)
        {
            int& pixelLabel = data[j] ;
            pixelLabel = labelSet[pixelLabel] ;
        }
    }
}
void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg)
{
    if (_labelImg.empty() ||
            _labelImg.type() != CV_32SC1)
    {
        return ;
    }

    std::map<int, cv::Scalar> colors ;

    int rows = _labelImg.rows ;
    int cols = _labelImg.cols ;

    _colorLabelImg.release() ;
    _colorLabelImg.create(rows, cols, CV_8UC3) ;
    _colorLabelImg = cv::Scalar::all(0) ;

    for (int i = 0; i < rows; i++)
    {
        const int* data_src = (int*)_labelImg.ptr<int>(i) ;
        uchar* data_dst = _colorLabelImg.ptr<uchar>(i) ;
        for (int j = 0; j < cols; j++)
        {
            int pixelValue = data_src[j] ;
            if (pixelValue > 1)
            {
                if (colors.count(pixelValue) <= 0)
                {
                    colors[pixelValue] = icvprGetRandomColor() ;
                }
                cv::Scalar color = colors[pixelValue] ;
                *data_dst++   = color[0] ;
                *data_dst++ = color[1] ;
                *data_dst++ = color[2] ;
            }
            else
            {
                data_dst++ ;
                data_dst++ ;
                data_dst++ ;
            }
        }
    }
}

int _tmain(int argc, char** argv)
{
	Mat image;
	image = imread("screenPrinting.png", 0); // Read the file
	cv::imshow("input image",image);
	
	//-----Transfer to frequency domain(code from openCV tutorial)----------------------------------
    Mat F,I;
    image.copyTo(I);
    I.convertTo(I,CV_32F);
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    cv::Mat mask=cv::Mat(padded.rows,padded.cols,CV_8UC1,cv::Scalar(255));
   
	int cx = mask.cols/2;
    int cy = mask.rows/2;
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    Mat magI,temp;
	magnitude(planes[0], planes[1], magI);// planes[0] = magnitude
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    cv::Mat ftmp;
    cx = magI.cols/2;
    cy = magI.rows/2;
    Mat f0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat f1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat f2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat f3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    f0.copyTo(ftmp);
    f3.copyTo(f0);
    ftmp.copyTo(f3);

    f1.copyTo(ftmp);                    // swap quadrant (Top-Right with Bottom-Left)
    f2.copyTo(f1);
    ftmp.copyTo(f2);
	 merge(planes, 2, complexI);
    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    // normalize(phaseVals, phaseVals, 0, 1, CV_MINMAX);
    // viewable image form (float between values 0 and 1).
    normalize(magI, magI, 0,255, CV_MINMAX); // Transform the matrix with float values into a
    magI.convertTo(magI,CV_8U);
	cv::imshow("frequency domain",magI);


	//-----------------put your filter here(threshold out smaller response frequency)--------------------------------------
    cv::threshold(magI, mask, 165, 255, CV_THRESH_BINARY) ;
    cv::line(mask,cv::Point(0,mask.rows/2),cv::Point(mask.cols,mask.rows/2),cv::Scalar(0),10);
    cv::line(mask,cv::Point(mask.cols/2,0),cv::Point(mask.cols/2,mask.rows),cv::Scalar(0),10);

    Mat q0(mask, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(mask, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(mask, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(mask, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    planes[0].copyTo(temp,mask);
    temp.copyTo(planes[0]);
    planes[1].copyTo(temp,mask);
    temp.copyTo(planes[1]);
    merge(planes, 2, complexI);


   //-------------Reconstruct image using only larger response in frequency domain----------------------------------------------------------------
    cv::Mat inverseTransform;
    cv::dft(complexI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

    normalize(inverseTransform, inverseTransform, 0, 255, CV_MINMAX);
    inverseTransform.convertTo(inverseTransform,CV_8U);
    imshow("Reconstructed",inverseTransform );
    inverseTransform.copyTo(image);
	
	//-------------find mesh in the reconstructed image-------------
    cv::Mat edge;
    image.copyTo(edge);
    cv::Mat otsu;
    cv::threshold(edge,otsu,150, 255,CV_THRESH_OTSU);
    otsu.convertTo(image,CV_8U);
    I.convertTo(I,CV_8U);
    vector <int> delta;
    for(int k=0;k<80;k++){ //shifting mesh to locate mesh with input image which has maximum value
        Scalar mean,std;
        cv::Mat out;
        Mat trans_mat = (Mat_<double>(2,3) << 1, 0, k, 0, 1, 0);
        warpAffine(image,out,trans_mat,image.size());
        cv::meanStdDev(I,mean,std,out);
        delta.push_back(mean.val[0]);
    }
    float max=0;
    int TH=0;
    for(int i=0;i<delta.size();i++)
        if(delta[i]>max){
            max = delta[i];
            TH=i;
        }
    cv::Mat out;
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, TH, 0, 1, 0);
    warpAffine(image,out,trans_mat,image.size());
    out=255-out;

	//-------------crop image for output visualization-------------
    int crop_w=600,crop_h=250;
    cv::Rect myROI(image.cols/2-crop_w/2, image.rows/2-crop_h/2, crop_w, crop_h);
    cv::Mat croppedImage = I(myROI);
	cv::imshow("croppedImage", croppedImage) ;
    warpAffine(edge,edge,trans_mat,image.size());
    cv::Mat croppedImage_2 = edge(myROI);
    cv::threshold(croppedImage_2,croppedImage_2,100, 255,CV_THRESH_OTSU);
    cv::threshold(croppedImage,croppedImage,100, 255,CV_THRESH_OTSU);
    croppedImage_2=255-croppedImage_2;
    croppedImage_2.copyTo(image);

	
	//------------using 4-connective method to segment different mesh-------------
    cv::threshold(image, image, 50, 1, CV_THRESH_BINARY) ;
    cv::Mat labelImg;
    int label=1;
    icvprCcaByTwoPass(image, labelImg); //label 4-connective area using two-pass methods
    cv::Mat grayImg ;
    cv::normalize(labelImg, grayImg, 0, 255,NORM_MINMAX);
    grayImg.convertTo(grayImg, CV_8UC1) ;
    grayImg.copyTo(image);
    cv::Mat colorLabelImg ;
    icvprLabelColor(labelImg, colorLabelImg) ; //using sudocolor to label different area
    cv::imshow("colorImg", colorLabelImg) ;
    colorLabelImg.copyTo(image);

	waitKey();
	return 0;
}

