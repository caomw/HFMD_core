#include "ctlkinect.h"

// setting file place
#define CONFIG_PATH "./xtionConfig.xml"

using namespace xn;

float* g_pDepthHist;
XnRGB24Pixel* g_pTexMap = NULL;
unsigned int g_nTexMapX = 0;
unsigned int g_nTexMapY = 0;
XnDepthPixel g_nZRes;

//unsigned int g_nViewState = DEFAULT_DISPLAY_MODE;

Context g_context;
ScriptNode g_scriptNode;
DepthGenerator g_depth;
ImageGenerator g_image;
DepthMetaData g_depthMD;
ImageMetaData g_imageMD;

CtlKinect::CtlKinect()
{
    //    // init kinect
    //    rc = XN_STATUS_OK;
    //    //open XML file
    //    rc = g_context.InitFromXmlFile(CONFIG_PATH);
    //    //rc = g_context.InitFromXmlFile(CONFIG_PATH, g_scriptNode, &errors);
    //    //if(rc != XN_STATUS_OK)
    //        //emit errorOccurred(0);

    //    //generate camear
    //    rc = g_context.FindExistingNode(XN_NODE_TYPE_IMAGE, g_image);
    ////    if(rc != XN_STATUS_OK)
    ////        emit errorOccurred(1);

    //    //generate depth
    //    rc = g_context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_depth);
    ////    if (rc != XN_STATUS_OK)
    ////        emit errorOccurred(2);

    //    XnMapOutputMode mapMode;

    //    //rc = g_depth.GetMapOutputMode(mapMode);
    ////    if (rc != XN_STATUS_OK)
    ////        emit errorOccurred(2);

    ////    qDebug() << tr("mapMode.nYRes is %1").arg(mapMode.nYRes);

    ////    rgb = new cv::Mat(mapMode.nYRes,mapMode.nXRes,CV_8UC3);
    ////    depth = new cv::Mat(mapMode.nYRes,mapMode.nXRes,CV_16UC1);
    ////    qDebug() << tr("%1 %1").arg(rgb->step,rgb->rows);


    //XnStatus rc;

    EnumerationErrors errors;
    rc = g_context.InitFromXmlFile(CONFIG_PATH, g_scriptNode, &errors);
    if (rc == XN_STATUS_NO_NODE_PRESENT)
    {
        XnChar strError[1024];
        errors.ToString(strError, 1024);
        printf("%s\n", strError);
        //return (rc);
    }
    else if (rc != XN_STATUS_OK)
    {
        printf("Open failed: %s\n", xnGetStatusString(rc));
        //return (rc);
    }

    rc = g_context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_depth);
    if (rc != XN_STATUS_OK)
    {
        printf("No depth node exists! Check your XML.");
        //return 1;
    }

    rc = g_context.FindExistingNode(XN_NODE_TYPE_IMAGE, g_image);
    if (rc != XN_STATUS_OK)
    {
        printf("No image node exists! Check your XML.");
        //return 1;
    }

    //        g_depth.GetMetaData(g_depthMD);
    //        g_image.GetMetaData(g_imageMD);

    //        // Hybrid mode isn't supported in this sample
    //        if (g_imageMD.FullXRes() != g_depthMD.FullXRes() || g_imageMD.FullYRes() != g_depthMD.FullYRes())
    //        {
    //            printf ("The device depth and image resolution must be equal!\n");
    //            return 1;
    //        }

    //        // RGB is the only image format supported.
    //        if (g_imageMD.PixelFormat() != XN_PIXEL_FORMAT_RGB24)
    //        {
    //            printf("The device image format must be RGB24\n");
    //            return 1;
    //        }

    //        // Texture map init
    //        g_nTexMapX = (((unsigned short)(g_depthMD.FullXRes()-1) / 512) + 1) * 512;
    //        g_nTexMapY = (((unsigned short)(g_depthMD.FullYRes()-1) / 512) + 1) * 512;
    //        g_pTexMap = (XnRGB24Pixel*)malloc(g_nTexMapX * g_nTexMapY * sizeof(XnRGB24Pixel));

    //        g_nZRes = g_depthMD.ZRes();
    //        g_pDepthHist = (float*)malloc(g_nZRes * sizeof(float));
}

CtlKinect::~CtlKinect(){
    //    delete rgb;
    //    delete depth;
}

void CtlKinect::getRGBDData(cv::Mat* rgb, cv::Mat* depth){
    //qDebug() << tr("start getting data");


    //cv::namedWindow("imagewindow");

    //while(1){
    rc = g_context.WaitAnyUpdateAll();
    //        if(rc != XN_STATUS_OK){
    //          emit errorOccurred(3);
    //          return;
    //        }

    //各種generatorからメタデータを取得しています
    g_depth.GetMetaData(g_depthMD);
    g_depth.GetAlternativeViewPointCap().SetViewPoint(g_image);
    g_image.GetMetaData(g_imageMD);


    //それぞれOpenCVの画像形式に変換しています
    memcpy(rgb->data,g_imageMD.Data(),rgb->step*rgb->rows);
    memcpy(depth->data,g_depthMD.Data(),depth->step*depth->rows);

    //そのままでは正しい色情報を持っていないので，OpenCVを使って変換してやります
    cv::cvtColor(*rgb,*rgb,CV_RGB2BGR);

    //cv::imshow("imagewindow", *rgb);

    //        emit getData(rgb, depth);
    //}

}







