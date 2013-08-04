#ifndef CTLKINECT_H
#define CTLKINECT_H

//#include <QObject>
//#include <QMessageBox>
//#include <QDebug>

#include <iostream>
#include <string>

//#ifndef Q_MOC_RUN
#include <XnOS.h>
#include <XnCppWrapper.h>
//#endif // Q_MOC_RUN
#include <opencv2/opencv.hpp>



#define CV_WIDTH 640
#define CV_HEIGHT 480


class CtlKinect// : public QObject
{
   // Q_OBJECT

public:
    CtlKinect();
    ~CtlKinect();
//public slots:
    void getRGBDData(cv::Mat*,cv::Mat*);

private:
    //cv::Mat *rgb, *depth;

    //Contextがすべての制御を担当します
    xn::Context g_context;
    //xn::ScriptNode g_scriptNode;

    XnStatus rc;

    //xn::EnumerationErrors errors;

    //Generatorクラスはそれぞれセンシングした値を返します
    xn::ImageGenerator g_image;
    xn::DepthGenerator g_depth;

    //MetaDataクラスはそれぞれ~Generatorクラスから得た情報を受け取ります
    xn::ImageMetaData g_imageMD;
    xn::DepthMetaData g_depthMD;

//signals:
//    void getData(cv::Mat*, cv::Mat*);
//    void errorOccurred(int);

};

#endif // CTLKINECT_H
