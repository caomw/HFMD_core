#include "CConfig.h"

using namespace boost;
using namespace boost::property_tree;

int CConfig::loadConfig(const char* filename)
{

  read_xml(filename, pt);

  try{



    // load tree path
    treepath = pt.get<std::string>("root.treepath");
    // load number of tree
    ntrees = pt.get<int>("root.ntree");


   

    // load patch width
    p_width = pt.get<int>("root.pwidth");
    // load patch height
    p_height = pt.get<int>("root.pheight");


    // load scale factor for output imae
    patchRatio = pt.get<double>("root.patchratio");
    // load patch stride
    stride = pt.get<int>("root.stride");
    // load train image num per tree
    imagePerTree = pt.get<int>("root.trainimagepertree");
    // load min sample num
    min_sample = pt.get<int>("root.minsample");
    // load max depth num
    max_depth = pt.get<int>("root.maxdepth");
    // load ratio of pos patch number and neg patch number
    pnRatio = pt.get<double>("root.posnegpatchratio");
    acPatchRatio = pt.get<double>("root.activepatchratio");
    nOfTrials = pt.get<int>("root.numberOfTrials");

  


    // learning mode
    // 1:depth 2:rgb 0:rgbd
    learningMode = pt.get<int>("root.learningmode");
    // rgb feature select
    // 0: haar-like, 1: HOG, 2: rotated haar-like
    rgbFeature = pt.get<int>("root.rgbfeature");
    // depth feature select
    // 0: haar-like, 1: HOG, 2: rotated haar-like
    depthFeature = pt.get<int>("root.depthfeature");

  
    

    // // load test image path
    // if (boost::optional<std::string>
    // 	= pt.get_optional<std::string>("root.imgpath")) {
    //   std::cout << str.get() << std::endl;
    //   impath = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load image name list
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.imgnamelist")) {
    //   std::cout << str.get() << std::endl;
    //   imfiles = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load extruct feature flag
    // if (boost::optional<bool> boolean
    // 	= pt.get_optional<bool>("root.efeatures")) {
    //   std::cout << boolean << std::endl;
    //   xtrFeature = *boolean;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // load image scales
    //std::cout << "kokomade" << std::endl;
    //    scales.resize(0);
    //    BOOST_FOREACH (const boost::property_tree::ptree::value_type& child,
    //                   pt.get_child("root.scales")) {
    //        const float value = boost::lexical_cast<float>(child.second.data());
    //        scales.push_back(value);

    //        std::cout << value << std::endl;
    //    }
    //    for (int i;i < scales.size(); ++i)
    //        std::cout << i << ": " << scales.at(i) << std::endl;
    //    float value_temp = 1;
    //    scales.clear();
    //    scales.push_back(value_temp);

    //    ratios.clear();
    //    ratios.push_back(value_temp);
    //    ratios.push_back(value_temp);

    // // load image ratios
    //    ratios.resize(0);
    //    BOOST_FOREACH (const boost::property_tree::ptree::value_type& child, pt.get_child("root.ratio")) {
    //        const float value = boost::lexical_cast<float>(child.second.data());
    //        ratios.push_back(value);

    //        std::cout << value << std::endl;
    //    }
    //    for (int i;i < ratios.size(); ++i)
    //        std::cout << i << ": " << ratios.at(i) << std::endl;

    // load output path
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.outpath")) {
    //   std::cout << str.get() << std::endl;
    //   outpath = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // load scale factor for output imae
    // if (boost::optional<int> integer
    // 	= pt.get_optional<int>("root.sfactor")) {
    //   std::cout << integer << std::endl;
    //   out_scale = *integer;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }


    // load training image name list
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.traindataname")) {
    //   std::cout << str.get() << std::endl;
    //   traindatafile = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load training image folder
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.trainimgpath")) {
    //   std::cout << str.get() << std::endl;
    //   trainpath = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    trainpath = pt.get<std::string>("root.trainposdata.rootpath");
    traindatafile = pt.get<std::string>("root.trainposdata.folderlist");
    traindatalist = pt.get<std::string>("root.trainposdata.imagelist");

    negDataPath = pt.get<std::string>("root.trainnegdata.rootpath");
    negFolderList = pt.get<std::string>("root.trainnegdata.folderlist");
    negDataList = pt.get<std::string>("root.trainnegdata.imagelist");
        
    testPath = pt.get<std::string>("root.testdata.rootpath");
    testData = pt.get<std::string>("root.testdata.folderlist");
    testdatalist = pt.get<std::string>("root.testdata.imagelist");


    off_tree = pt.get<int>("root.offtree");
    classDatabaseName = pt.get<int>("root.classdatabasename");

    mindist = pt.get<int>("root.mindistance");
    maxdist = pt.get<int>("root.maxdistance");

    

    // // load scale factor for output imae
    // if (boost::optional<int> integer
    // 	= pt.get_optional<int>("root.featurechannel")) {
    //   std::cout << integer << std::endl;
    //   featureChannel = *integer;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }



    // // load offset of tree name
    // if (boost::optional<int> integer
    // 	= pt.get_optional<int>("root.offTree")) {
    //   std::cout << integer << std::endl;
    //   off_tree = *integer;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load test data folder
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.testpath")) {
    //   std::cout << str.get() << std::endl;
    //   testPath = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load test data file name
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.testdataname")) {
    //   std::cout << str.get() << std::endl;
    //   testData = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load test data file name
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.classdatabasename")) {
    //   std::cout << str.get() << std::endl;
    //   classDatabaseName = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load offset of tree name
    // if (boost::optional<int> integer
    // 	= pt.get_optional<int>("root.learningmode")) {
    //   std::cout << "learning mode is " << integer << std::endl;
    //   learningMode = *integer;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load training image name list
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.traindatalistname")) {
    //   std::cout << str.get() << std::endl;
    //   traindatalist = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load testing image name list
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.testdatalistname")) {
    //   std::cout << str.get() << std::endl;
    //   testdatalist = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load offset of tree name
    // if (boost::optional<int> integer
    // 	= pt.get_optional<int>("root.testmode")) {
    //   std::cout << integer << std::endl;
    //   testMode = *integer;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load testing image name list
    // if (boost::optional<double> dou
    // 	= pt.get_optional<double>("root.detectthreshold")) {
    //   std::cout << dou.get() << std::endl;
    //   detectThreshold = *dou;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load offset of tree name
    // if (boost::optional<int> integer
    // 	= pt.get_optional<int>("root.showgrandtruth")) {
    //   std::cout << integer << std::endl;
    //   showGT = *integer;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load training image folder
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.negativedatapath")) {
    //   std::cout << str.get() << std::endl;
    //   negDataPath = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load training image folder
    // if (boost::optional<std::string> str
    // 	= pt.get_optional<std::string>("root.negativedatalist")) {
    //   std::cout << str.get() << std::endl;
    //   negDataList = *str;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load offset of tree name
    // if (boost::optional<int> integer
    // 	= pt.get_optional<int>("root.negativemode")) {
    //   std::cout << integer << std::endl;
    //   negMode = *integer;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load testing image name list
    // if (boost::optional<double> dou
    // 	= pt.get_optional<double>("root.posnegpatchratio")) {
    //   std::cout << dou.get() << std::endl;
    //   pnRatio = *dou;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load testing image name list
    // if (boost::optional<double> dou
    // 	= pt.get_optional<double>("root.activepatchratio")) {
    //   std::cout << dou.get() << std::endl;
    //   acPatchRatio = *dou;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // // load testing image name list
    // if (boost::optional<double> dou
    // 	= pt.get_optional<double>("root.mindistance")) {
    //   std::cout << dou.get() << std::endl;
    //   mindist = *dou;
    // }
    // else {
    //   std::cout << "root.str is nothing" << std::endl;
    // }

    // maxdist = *pt.get_optional<double>("root.maxdistance");

    // nOfTrials = *pt.get_optional<int>("root.numberOfTrials");

    // negFolderList = *pt.get_optional<std::string>("root.negativedatafolderlist");

    // crossVMode = *pt.get_optional<int>("root.crossvalidationmode");

    // clusterNumLimit = *pt.get_optional<int>("root.clusterNumLimit");

    // widthScale = p_width / mindist;
    // heightScale = p_height / mindist;

    // rgbFeature = *pt.get_optional<int>("root.rgbFeature");
    // depthFeature = *pt.get_optional<int>("root.depthFeature");

    // paramRadius = *pt.get_optional<int>("root.paramradius");

    // modelListFolder = *pt.get_optional<std::string>("root.modellistfolder");
    // modelListName = *pt.get_optional<std::string>("root.modellistname");

    // modelLearningMode = *pt.get_optional<int>("root.modellearningmode");

  }

  catch(ptree_bad_path bp){
    std::cout << "bad property tree path! " << bp.path<std::string>() << std::endl;
  }
  catch(ptree_bad_data bd){
    std::cout << "bad property tree data! " << bd.data<std::string>() << std::endl;
  }

  return 0;
}

