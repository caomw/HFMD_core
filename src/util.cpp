#include "util.h"

boost::lagged_fibonacci1279 nCk::gen = boost::lagged_fibonacci1279();
boost::lagged_fibonacci1279 genPose = boost::lagged_fibonacci1279();

float calcSumOfDepth(cv::Mat &depth, const CConfig &conf){
  cv::Mat convertedDepth = cv::Mat(depth.rows, depth.cols, CV_8U);
  cv::Mat integralMat = cv::Mat(depth.rows + 1, depth.cols+1, CV_32F);
  cv::Mat temp1,temp2;
  depth.convertTo(convertedDepth,CV_8U,255.0/(double)(conf.maxdist - conf.mindist));

  cv::integral(convertedDepth,integralMat,temp1,temp2,CV_32F);
  return integralMat.at<int>(depth.rows, depth.cols);
}

void loadTrainObjFile(CConfig conf, std::vector<CPosDataset*> &posSet)
{
  std::vector<std::string> modelPath(0);
  std::vector<std::string> modelName(0);
  std::string trainModelListPath = conf.modelListFolder + PATH_SEP + conf.modelListName;

  boost::uniform_real<> dst(0, 360);
  boost::variate_generator<boost::lagged_fibonacci1279&,
			   boost::uniform_real<> > rand(genPose, dst);

  posSet.clear();

  std::ifstream modelList(trainModelListPath.c_str());
  if(!modelList.is_open()){
    std::cout << "train model list is not found!" << std::endl;
    exit(-1);
  }

  int modelNum = 0;
  modelList >> modelNum;
  modelPath.resize(modelNum);
  modelName.resize(modelNum);

  for(int i = 0; i < modelNum; ++i){
    std::string tempName;
    modelList >> tempName;
    modelPath[i] = conf.modelListFolder +PATH_SEP + tempName;
    std::string tempClass;
    modelList >> tempClass;
    modelName[i] = tempClass;
  }

  posSet.resize(modelNum * conf.imagePerTree);
    
  //    std::cout << modelNum << std::endl;
  for(int j = 0; j < modelNum; ++j){
    for(int i = 0; i < conf.imagePerTree; ++i){
      CPosDataset* posTemp = new CPosDataset();
      posTemp->setModelPath(modelPath.at(j));
      posTemp->setClassName(modelName.at(j));
      double tempAngle[3];
      for(int k = 0; k < 3; ++k)
	tempAngle[k] = rand();
      posTemp->setAngle(tempAngle);
      posTemp->setCenterPoint(cv::Point(320,240));

      posSet[j * conf.imagePerTree + i] = posTemp;
    }
  }
  modelList.close();
}

void loadTrainPosFile(CConfig conf, std::vector<CPosDataset*> &posSet)
{
  std::string traindatafilepath = conf.trainpath + PATH_SEP +  conf.traindatafile;
  int n_folders;
  int n_files;
  std::vector<std::string> trainimagefolder;
  std::vector<CPosDataset*> tempDataSet;
  std::string trainDataListPath;
  //int dataSetNum = 0;
  CClassDatabase database;
  cv::Point tempPoint;
  nCk nck;

  posSet.clear();

  //read train data folder list
  std::ifstream in(traindatafilepath.c_str());
  if(!in.is_open()){
    std::cout << "train data floder list " << traindatafilepath.c_str() << " is not found!" << std::endl;
    exit(1);
  }

  // read train pos folder
  in >> n_folders;
  std::cout << "number of training data folders : " << n_folders << std::endl;
  trainimagefolder.resize(n_folders);
  for(int i = 0;i < n_folders; ++i)
    in >> trainimagefolder.at(i);
  std::cout << trainimagefolder.at(0) << std::endl;

  // close train pos data folder list
  in.close();

  std::cout << "train folders lists : " << std::endl;
  //read train file name and grand truth from file
  tempDataSet.clear();
  for(int i = 0;i < n_folders; ++i){
    trainDataListPath
      = conf.trainpath + PATH_SEP + trainimagefolder.at(i) + PATH_SEP + conf.traindatalist;

    std::cout << trainDataListPath << std::endl;
    std::string imageFilePath
      = conf.trainpath + PATH_SEP + trainimagefolder.at(i) + PATH_SEP;

    std::ifstream trainDataList(trainDataListPath.c_str());
    if(!trainDataList.is_open()){
      std::cout << "can't read " << trainDataListPath << std::endl;
      exit(1);
    }

    trainDataList >> n_files;

    for(int j = 0;j < n_files; ++j){
      CPosDataset* posTemp = new CPosDataset();
      std::string nameTemp;

      //read file names
      trainDataList >> nameTemp;
      posTemp->setRgbImagePath(imageFilePath + nameTemp);

      trainDataList >> nameTemp;
      posTemp->setDepthImagePath(imageFilePath + nameTemp);

      trainDataList >> nameTemp;// dummy

      //read class name
      std::string tempClassName;
      trainDataList >> tempClassName;
      posTemp->setClassName(tempClassName);

      // read image size
      cv::Mat tempImage = cv::imread(posTemp->getRgbImagePath(),3);
      cv::Size tempSize = cv::Size(tempImage.cols, tempImage.rows);

      // set center point
      tempPoint = cv::Point(tempImage.cols / 2, tempImage.rows / 2);
      posTemp->setCenterPoint(tempPoint);

      // registre class param to class database
      database.add(posTemp->getParam()->getClassName(), tempSize, 0);

      //read angle grand truth
      double tempAngle[3];
      for(int i = 0; i < 3; ++i)
	trainDataList >> tempAngle[i];

      posTemp->setAngle(tempAngle);

      tempDataSet.push_back(posTemp);
    }

    trainDataList.close();
  }

  int dataOffset = 0;
  database.show();
  for(unsigned int j = 0;j < database.vNode.size(); j++){
    std::set<int> chosenData = nck.generate(database.vNode.at(j).instances, conf.imagePerTree);

    std::set<int>::iterator ite = chosenData.begin();

    while(ite != chosenData.end()){
      posSet.push_back(tempDataSet.at(*ite + dataOffset));
      //std::cout << tempDataSet.at(*ite + dataOffset).getRgbImagePath() << std::endl;
      ite++;
    }
    dataOffset += database.vNode.at(j).instances;
  }
}

void loadTrainNegFile(CConfig conf, std::vector<CNegDataset*> &negSet)
{
  //std::string traindatafilepath
  int n_folders;
  std::ifstream in_F((conf.negDataPath + PATH_SEP + conf.negFolderList).c_str());
  in_F >> n_folders;
  std::cout << conf.negDataPath + PATH_SEP + conf.negFolderList << std::endl;


  for(int j = 0; j < n_folders; ++j){
    int n_files;
    CDataset temp;
    //std::string trainDataListPath = conf.negDataPath + PATH_SEP +  conf.negDataList;

    std::string negDataFolderName;
    in_F >> negDataFolderName;
    std::string negDataFilePath = conf.negDataPath + PATH_SEP + negDataFolderName +  PATH_SEP;

    //read train data folder list
    std::ifstream in((negDataFilePath + conf.negDataList).c_str());
    if(!in.is_open()){
      std::cout << negDataFilePath << " train negative data floder list is not found!" << std::endl;
      exit(1);
    }

    std::cout << negDataFilePath << " loaded!" << std::endl;
    in >> n_files;

    //negSet.clear();

    for(int i = 0; i < n_files; ++i){
      CNegDataset* negTemp = new CNegDataset();
      std::string tempName;

      in >> tempName;
      negTemp->setRgbImagePath(negDataFilePath + tempName);

      in >> tempName;
      negTemp->setDepthImagePath(negDataFilePath + tempName);

      negSet.push_back(negTemp);
    }

    in.close();
  }

  in_F.close();
  //    for(int i = 0; i < dataSet.size(); ++i)
  //        dataSet.at(i).showDataset();
}

// extract patch from images
// !!!!!!coution!!!!!!!
// this function is use for negatime mode!!!!!
void extractPosPatches(std::vector<CPosDataset*> &posSet,
                       std::vector<CPosPatch> &posPatch,
                       CConfig conf,
                       const int treeNum,
                       const CClassDatabase &classDatabase){
  cv::Rect tempRect;
  //  std::cout << "kokokara" << std::endl;
  std::vector<CPosPatch> tPosPatch(0);//, posPatch(0);
  std::vector<std::vector<CPosPatch> > patchPerClass(classDatabase.vNode.size());
  //int pixNum;
  nCk nck;
  int classNum = 0;
  cv::Mat roi;

  // cv::Mat showDepth = cv::Mat(posSet[0]->img[1]->rows, posSet[0]->img[1]->cols, CV_8U);
  // posSet[0]->img[1]->convertTo(showDepth, CV_8U, 255.0/1000.0);
    
  // cv::namedWindow("test");
  //    cv::imshow("test", *posSet[0]->img[0]);
  //    cv::namedWindow("test2");
  //    cv::imshow("test2", showDepth);

  //    cv::waitKey(0);

  //    cv::destroyWindow("test");
  //    cv::destroyWindow("test2");


  //    std::cout << posSet[1]->img[1]->type() << " " << CV_16U << std::endl;

  posPatch.clear();

  tempRect.width  = conf.p_width;
  tempRect.height = conf.p_height;
  

  std::cout << "image num is " << posSet.size() << std::endl;;

  std::cout << "extracting pos patch from image" << std::endl;
  for(unsigned int l = 0;l < posSet.size(); ++l){

    for(int j = 0; j < posSet.at(l)->img.at(0)->cols - conf.p_width; j += conf.stride){
      for(int k = 0; k < posSet.at(l)->img.at(0)->rows - conf.p_height; k += conf.stride){
	tempRect.x = j;
	tempRect.y = k;

	// set patch class
	classNum = classDatabase.search(posSet.at(l)->getParam()->getClassName());//dataSet.at(l).className.at(0));
	if(classNum == -1){
	  std::cout << "class not found!" << std::endl;
	  exit(-1);
	}

	//tPatch.setPatch(temp, image.at(l), dataSet.at(l), classNum);

	CPosPatch posTemp(tempRect, posSet.at(l));

	int centerDepthFlag = 0;

	if(conf.learningMode != 2){
	  cv::Mat tempDepth1 = *posTemp.getDepth();
	  cv::Mat tempDepth2 = tempDepth1(tempRect);

	  // if()
	  //   centerDepthFlag = 1;
	  //std::cout << tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) << std::endl;
	
	


	  //	std::cout << centerDepthFlag << std::endl;

	  //if (conf.learningMode == 2){// || pixNum > 0){

	  //std::cout << tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) << std::endl;
	  if(tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) != 0){
	    //	  std::cout << "test" << std::endl;
	    //if(conf.learningMode != 2){
	    normarizationByDepth(posTemp , conf, (double)tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1));
	    normarizationCenterPointP(posTemp, conf, (double)tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1));
	    //}

	  
	    //	  std::cout << "kokomade" << std::endl;
	    //std::cout << posTemp.getRoi().width << std::endl;

	  }
	}
	if(posTemp.getRoi().width > 5 && posTemp.getRoi().height > 5){
	  std::vector<double> pRatio(0);
	  // pRatio.push_back(1.0);
	  // pRatio.push_back(1.4);
	  // pRatio.push_back(1.8);
	  // pRatio.push_back(2.2);
	  // pRatio.push_back(2.6);
	  // pRatio.push_back(3.0);
	  // pRatio.push_back(3.4);
	  // pRatio.push_back(3.8);
	  // pRatio.push_back(4.2);
	  // pRatio.push_back(4.6);
	  // pRatio.push_back(5.0);

	  pRatio.push_back(1.0);
	  pRatio.push_back(1.2);
	  pRatio.push_back(1.4);
	  pRatio.push_back(1.6);
	  pRatio.push_back(1.8);
	  pRatio.push_back(2.0);

	  for(uint r = 0; r < pRatio.size(); ++r){
	    CPosPatch transPatch(posTemp);
	    cv::Rect tempRoi = transPatch.getRoi();
	    

	    if(tempRoi.x - (int)((double)tempRoi.width * (pRatio[r] - 1.0) / 2.0)> 0)
	      tempRoi.x -= (int)((double)tempRoi.width * (pRatio[r] - 1.0) / 2.0);
	    else
	      tempRoi.x = 0;

	    if(tempRoi.y - (int)((double)tempRoi.height * (pRatio[r] - 1.0) / 2.0)> 0)
	      tempRoi.y -= (int)((double)tempRoi.height * (pRatio[r] - 1.0) / 2.0);
	    else
	      tempRoi.y = 0;

	    if(tempRoi.width * pRatio[r] + tempRoi.x < posSet.at(l)->img.at(0)->cols)
	      tempRoi.width *= pRatio[r];
	    else
	      tempRoi.width = posSet.at(l)->img.at(0)->cols - tempRoi.x;

	    if(tempRoi.height * pRatio[r] + tempRoi.y < posSet.at(l)->img.at(0)->rows)
	      tempRoi.height *= pRatio[r];
	    else
	      tempRoi.height = posSet.at(l)->img.at(0)->rows - tempRoi.y;

	    transPatch.setRoi(tempRoi);
	    
	    cv::Point_<double> tempRP = transPatch.getRelativePosition();

	    tempRP.x /= pRatio[r];
	    tempRP.y /= pRatio[r];

	    transPatch.setRelativePosition(tempRP);

	    // tPosPatch.push_back(posTemp);
	    // patchPerClass.at(classNum).push_back(posTemp);

	    tPosPatch.push_back(transPatch);
	    patchPerClass.at(classNum).push_back(transPatch);

	  }
	}

	//} // if
      }//x
    }//y
  }//allimages

  // for(unsigned int w = 0; w < patchPerClass.size(); ++w){
  //   std::cout << patchPerClass.at(w).size() << " ";
  // }
  // std::cout << std::endl;

  std::vector<int> patchNum(patchPerClass.size(),conf.patchRatio);

  for(unsigned int i = 0; i < patchPerClass.size(); ++i){
    if(i == treeNum % patchPerClass.size())
      patchNum.at(i) = conf.patchRatio;
    else
      patchNum.at(i) = conf.patchRatio * conf.acPatchRatio;
  }

  // for(unsigned int w = 0; w < patchPerClass.size(); ++w){
  //   std::cout << patchPerClass.at(w).size() << " ";
  // }

  // choose patch from each image
  for(unsigned int i = 0; i < patchPerClass.size(); ++i){
    if(patchPerClass.at(i).size() > conf.patchRatio){

      std::set<int> chosenPatch = nck.generate(patchPerClass.at(i).size(),patchNum.at(i));// conf.patchRatio);//totalPatchNum * conf.patchRatio);
      std::set<int>::iterator ite = chosenPatch.begin();

      while(ite != chosenPatch.end()){
	//std::cout << "this is for debug ite is " << tPosPatch.at(*ite).center << std::endl;
	//std::cout <<posPatch.at(i)
	//std::cout << patchPerClass.at(i).at(*ite).getRgbImageFilePath() << std::endl;
	posPatch.push_back(patchPerClass.at(i).at(*ite));
	ite++;
      }
    }else{
      //                std::cout << classNum << std::endl;
      ////            cv::namedWindow("test");
      ////            cv::imshow("test", *posSet.at(0).img.at(0));
      ////            cv::waitKey(0);
      ////            cv::destroyWindow("test");
      //            //std::cout << *posSet.at(1).img.at(1) << std::endl;
      //                std::cout << patchPerClass.size() << std::endl;

      std::cout << "can't extruct enough patch" << std::endl;
      //exit(-1);
    }
  }
  return;
}

void extractNegPatches(std::vector<CNegDataset*> &negSet,
                       std::vector<CNegPatch> &negPatch,
                       CConfig conf){
  cv::Rect tempRect;

  std::vector<CNegPatch> tNegPatch(0);//, posPatch(0);
  nCk nck;
  unsigned int negPatchNum = 0;

  negPatch.clear();

  tempRect.width  = conf.p_width;
  tempRect.height = conf.p_height;

  // extract negative patch
  //std::cout << negSet.size() << std::endl;
  //std::cout << negSet.at(0)->img.at(0)->cols << std::endl;
  //std::cout << negSet.at(0)->img.at(0)->rows << std::endl;
  for(unsigned int i = 0; i < negSet.size(); ++i){
    for(int j = 0; j < negSet.at(i)->img.at(0)->cols - conf.p_width; j += conf.stride){
      for(int k = 0; k < negSet.at(i)->img.at(0)->rows - conf.p_height; k += conf.stride){

	tempRect.x = j;
	tempRect.y = k;
	//int pix = 0;

	// detect negative patch
	//                if(conf.learningMode != 2 && negSet.at(i).img.at(1)->at<ushort>(k + (conf.p_height / 2) + 1, j + (conf.p_width / 2) + 1 ) != 0){
	//                    cv::Mat roi;
	//                    roi = (*negSet.at(i).img.at(1))(cv::Rect(j, k, conf.p_width, conf.p_height));
	//                    pix = calcSumOfDepth(roi,conf);
	//                }


	//if (conf.learningMode == 2){// || pixNum > 0){
	//if(centerDepthFlag != 1){

	//tPatch.setPatch(temp, negImage.at(i));
	//if(conf.learningMode == 2 || pix > 0){
	CNegPatch negTemp(tempRect, negSet.at(i));

	int centerDepthFlag = 0;

	if(conf.learningMode != 2){
	  cv::Mat tempDepth1 = *negTemp.getDepth();
	  cv::Mat tempDepth2 = tempDepth1(tempRect);

	  //	  if( != 0)
	  // centerDepthFlag = 1;

	
	  if(tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) != 0){
	    //if(conf.learningMode)
	    normarizationByDepth(negTemp , conf,(double)tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1));


	  }
	}

	if(negTemp.getRoi().width > 5 && negTemp.getRoi().height > 5)
	  tNegPatch.push_back(negTemp);
	//}
      }//x
    }//y
  } // every image

    // choose negative patch randamly
  negPatchNum = conf.patchRatio * conf.pnRatio;
  //std::cout << "pos patch num : " << posPatch.size() << " neg patch num : " << negPatchNum << std::endl;

  if(negPatchNum < tNegPatch.size()){
    std::set<int> chosenPatch = nck.generate(tNegPatch.size(), negPatchNum);//totalPatchNum * conf.patchRatio);
    std::set<int>::iterator ite = chosenPatch.begin();

    while(ite != chosenPatch.end()){
      negPatch.push_back(tNegPatch.at(*ite));
      ite++;
    }
  }else{
    std::cout << "only " << tNegPatch.size() << " pathes extracted from negative images" << std::endl;
    std::cout << "can't extract enogh negative patch please set pnratio more low!" << std::endl;
    exit(-1);
  }

  //    patches.push_back(posPatch);
  //    patches.push_back(negPatch);
}

void extractTestPatches(CTestDataset* testSet,std::vector<CTestPatch> &testPatch, CConfig conf){

  cv::Rect tempRect;
  //CPatch tPatch;




  // double widthSca = (double)(testSet->img[1]->at<ushort>(240 , 320));
  // double heightSca = (double)(testSet->img[1]->at<ushort>(240, 320));



  // cv::Mat test = testSet->img[1]->clone();
  // cv::Mat showTest;
  // int face[] = {cv::FONT_HERSHEY_SIMPLEX, cv::FONT_HERSHEY_PLAIN, cv::FONT_HERSHEY_DUPLEX, cv::FONT_HERSHEY_COMPLEX, 
  //               cv::FONT_HERSHEY_TRIPLEX, cv::FONT_HERSHEY_COMPLEX_SMALL, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 
  //               cv::FONT_HERSHEY_SCRIPT_COMPLEX, cv::FONT_ITALIC};

  // std::cout << widthSca << " " << heightSca << std::endl;

  // if(widthSca > 0 && heightSca > 0)
  //   cv::resize(test, test, cv::Size(), widthSca / 1000.0, heightSca / 1000.0);

  // test.convertTo(showTest, CV_8UC1, 255.0 / 1000);

  // std::stringstream ss;
  // ss << widthSca;
  // cv::putText(showTest, ss.str(), cv::Point(320,240), face[0], 1.2, cv::Scalar(0,0,200), 2, CV_AA);

  // cv::imshow("test",showTest);
  // cv::waitKey(1);


  tempRect.width = conf.p_width;
  tempRect.height = conf.p_height;

  int imgCol = testSet->img[0]->cols;
  int imgRow = testSet->img[0]->rows;
  
  testPatch.clear();
  testPatch.reserve((int)(((double)imgCol / (double)conf.stride) * ((double)imgRow / (double)conf.stride)));
  //std::cout << "extraction patches!" << std::endl;
  for(int j = 0; j < imgCol - conf.p_width; j += conf.stride){
    for(int k = 0; k < imgRow - conf.p_height; k += conf.stride){
      tempRect.x = j;
      tempRect.y = k;

      //int depthSum;

      //            if(conf.learningMode != 2 && testSet.img.at(1)->at<ushort>(k + (conf.p_height / 2) + 1, j + (conf.p_width / 2) + 1 ) != 0){
      //                cv::Mat roi;
      //                roi = (*testSet.img.at(1))(cv::Rect(j, k, conf.p_width, conf.p_height));
      //                depthSum = calcSumOfDepth(roi,conf);
      //            }
      //                for(int p = 0; p < testSet.feature.at(32)->rows; ++p)
      //                    for(int q = 0; q < testSet.feature.at(32)->cols; ++q)
      //                        depthSum += testSet.feature.at(32)->at<ushort>(p, q);

      //std::cout << dataSet.className << std::endl;

      //int classNum = classDatabase.search(dataSet.className.at(0));

      //std::cout << "class num is " << classNum << std::endl;
      //classDatabase.show();
      //if(classNum == -1){
      //std::cout << "This tree not contain this class data" << std::endl;
      //exit(-1);
      //}

      //if(conf.learningMode == 2){// || depthSum > 0){
      CTestPatch testTemp(tempRect, testSet);
      //tesetPatch.setPatch(temp, image, dataSet, classNum);


      int centerDepthFlag = 0;

      if(conf.learningMode != 2){
	cv::Mat tempDepth1 = *testTemp.getDepth();
	cv::Mat tempDepth2 = tempDepth1(tempRect);

	if(tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) != 0)
	  centerDepthFlag = 1;

	//}

	//tPatch.setPosition(j,k);
	if(tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) != 0)//centerDepthFlag){
	  if(testSet->img.size() > 1)
	    normarizationByDepth(testTemp , conf, (double)tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1));


      }
      if(testTemp.getRoi().width > 5 && testTemp.getRoi().height > 5)
	testPatch.push_back(testTemp);
    
      //}
    }
  }
  

}

void pBar(int p,int maxNum, int width){
  int i;
  std::cout << "[";// << std::flush;
  for(i = 0;i < (int)((double)(p + 1)/(double)maxNum*(double)width);++i)
    std::cout << "*";

  for(int j = 0;j < width - i;++j)
    std::cout << " ";

  std::cout << "]" << (int)((double)(p + 1)/(double)maxNum*100) << "%"  << "\r" << std::flush;
}

void CClassDatabase::add(std::string str, cv::Size size, uchar depth){
  for(unsigned int i = 0; i < vNode.size(); ++i){
    if(str == vNode.at(i).name){
      vNode.at(i).instances++;
      return;
    }
  }
  //std::cout << str << " " << size << " " << depth << std::endl;
  vNode.push_back(databaseNode(str,size,depth));
  return;
}

void CClassDatabase::write(const char* str){

  std::ofstream out(str);
  if(!out.is_open()){
    std::cout << "can't open " << str << std::endl;
    return;
  }

  for(unsigned int i = 0; i < vNode.size(); ++i){
    out << i << " " << vNode.at(i).name
	<< " " << vNode.at(i).classSize.width << " " << vNode.at(i).classSize.height
	<< " " << vNode.at(i).classDepth
	<< std::endl;
  }
  out.close();

  //  std::cout << "out ha shimemashita" << std::endl;
}

void CClassDatabase::read(const char* str){
  std::string tempStr;
  std::stringstream tempStream;
  int tempClassNum;
  std::string tempClassName;
  cv::Size tempSize;
  uchar tempDepth;


  std::ifstream in(str);
  if(!in.is_open()){
    std::cout << "can't open " << str << std::endl;
    return;
  }

  //std::cout << str << std::endl;

  //vNode.clear();

  do{
    in >> tempClassNum;
    in >> tempClassName;
    in >> tempSize.width;
    in >> tempSize.height;
    in >> tempDepth;
    in.ignore();
    if(!in.eof())
      this->add(tempClassName,tempSize,tempDepth);

  }while(!in.eof());

  in.close();
}

void CClassDatabase::show() const{
  if(vNode.size() == 0){
    std::cout << "No class registerd" << std::endl;
    return;
  }

  for(unsigned int i = 0; i < vNode.size(); ++i){
    std::cout << "class:" << i << " name:" << vNode.at(i).name << " has " << vNode.at(i).instances << " instances" << std::endl;
  }
}

int CClassDatabase::search(std::string str) const{
  for(unsigned int i = 0; i < vNode.size(); i++){
    //std::cout << i << " " << str << " " << vNode.at(i).name << std::endl;
    if(str == vNode.at(i).name)return i;
  }
  return -1;
}

int normarizationByDepth(CPatch &patch, const CConfig &config, double sca){//, const CConfig &config)const {
  cv::Mat tempDepth = *patch.getDepth();
  cv::Mat depth = tempDepth(patch.getRoi());

  //double sca = tempDepth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1);

  cv::Rect roi;

  roi.width = patch.getRoi().width * config.maxdist / sca;
  roi.height = patch .getRoi().height * config.maxdist / sca;

  roi.x = patch.getRoi().x - roi.width / 2;
  roi.y = patch.getRoi().y - roi.height / 2;
  //std::cout << patch.getRoi() << std::endl;

  //  cv::namedWindow
   

  //roi.width = (int)realWindowWidth;
  //roi.height = (int)realWindowHeight;

  //std::cout << roi.x << " " << roi.y << " " << roi.width << " "<< roi.height << std::endl;

  //std::cout << depth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1) << std::endl;
  //std::cout << roi << std::endl;


  //  std::cout << roi << std::endl;

  //std::cout << patch.getDepth()->rows << std::endl;

  //  std::cout << roi << std::endl;
  if(roi.x < 0) roi.x = 0;
  if(roi.y < 0) roi.y = 0;
  if(roi.x + roi.width > patch.getDepth()->cols) roi.width = patch.getDepth()->cols - roi.x;
  if(roi.y + roi.height > patch.getDepth()->rows) roi.height = patch.getDepth()->rows - roi.y;

  //std::cout << sca << std::endl;

  //std::cout << tempDepth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1) << std::endl;
  //std::cout << roi << std::endl;
  //std::cout << "normarized roi is ";
  //std::cout << roi << std::endl;
  //rgb = rgb(roi);
  patch.setRoi(roi);
  //cv::resize(rgb,rgb,cv::Size(config.p_width,config.p_height));


  //                cv::namedWindow("test");
  //                cv::imshow("test",rgb);
  //                cv::waitKey(0);
  //                cv::destroyAllWindows();
  //    std::cout << "kokoke-" << std::endl;
  return 0;
}

int normarizationCenterPointP(CPosPatch &patch, const CConfig &config, double sca){//, const CConfig &config)const {
  cv::Mat tempDepth = *patch.getDepth();
  cv::Mat depth = tempDepth(patch.getRoi());


  
  //cv::Mat showDepth = cv::Mat(tempDepth.rows, tempDepth.cols, CV_8UC1);

  //tempDepth.convertTo(showDepth, CV_8UC1, 255 / 1000);

  //cv::namedWindow("test");
  //cv::imshow("test", showDepth);
  //cv::waitKey(0);
  //cv::destroyWindow("test");

  //calc width and height scale
  //std::cout << depth.type() << " " << CV_8U << std::endl;
  //std::cout << config.p_height / 2 + 1 <<  config.p_width / 2 + 1 << std::endl;
  //  std::cout << "depth rows and cols " << depth.rows << " " << depth.cols << std::endl;
  //double centerDepth = depth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1) + config.mindist;
  cv::Point_<double> currentP = patch.getRelativePosition();

  //  double sca = tempDepth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1);
  //  if(sca == 0)
  //  exit(-1);
  //std::cout << "current p before " << currentP << std::endl;
  //std::cout << sca << std::endl;

  //    currentP.x = currentP.x * 10;
  //currentP.y *= 1000;
  //currentP.x *= 1000;
  currentP.x = currentP.x * 100 / sca;
  currentP.y = currentP.y * 100 / sca;

  //std::cout << "current p " << currentP << std::endl;

  //std::cout << "kokomade" << std::endl;
  //  std::cout << "heknak go " << currentP << std::endl;
  patch.setRelativePosition(currentP);

  return 0;
}

