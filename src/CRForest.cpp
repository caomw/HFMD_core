#include <boost/timer.hpp>
//#include <boost/timer/timer.hpp>
#include "CRForest.h"

paramHist& paramHist::operator +(const paramHist& obj){ 
  this->roll += obj.roll;
  this->pitch += obj.pitch;
  this->yaw += obj.yaw;

  return *this;
}

paramHist& paramHist::operator +=(const paramHist& obj){
  this->roll += obj.roll;
  this->pitch += obj.pitch;
  this->yaw += obj.yaw;

  return *this;
}

void paramHist::showHist(){
  for(int i = 0; i < 360; ++i){
    std::cout << i << " degree " << yaw.at<double>(0,i) <<  std::endl;
  }
}

double euclideanDist(cv::Point p, cv::Point q)
{
  cv::Point diff = p - q;
  return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

void CRForest::learning(){
  //#pragma omp parallel
  //    {
  //#pragma omp for
  for(int i = 0;i < conf.ntrees; ++i){
    growATree(i);
  } // end tree loop
    //}
}

void CRForest::growATree(const int treeNum){
  // positive, negative dataset
  std::vector<CPosDataset> posSet(0);
  std::vector<CNegDataset> negSet(0);

  // positive, negative patch
  std::vector<CPosPatch> posPatch(0);
  std::vector<CNegPatch> negPatch(0);

  char buffer[256];

  std::cout << "tree number " << treeNum << std::endl;

  // initialize random seed
  boost::mt19937    gen( treeNum * static_cast<unsigned long>(time(NULL)) );
  //boost::timer t;


  //loadTrainPosFile(conf, posSet);//, gen);
  if(conf.modelLearningMode)
    loadTrainObjFile(conf,posSet);
  else
    loadTrainPosFile(conf, posSet);

  CClassDatabase tempClassDatabase;
  // extract pos features and register classDatabase
  for(int i = 0; i < posSet.size(); ++i){
    //std::cout << i << std::endl;

    //std::cout << posSet.at(i).rgb << std::endl;
    //        if(posSet.at(i).loadImage(conf) == -1 && conf.learningMode != 2){
    //            exit(-1);
    //        }

    //posSet.at(i).extractFeatures(conf);

    //std::cout << posSet.size() << std::endl;

    //tempClassDatabase.show();
    //cv::Mat test = cv::Mat(640,480);
    //    tempClassDatabase.add(

    
    tempClassDatabase.add(posSet.at(i).getParam()->getClassName(),cv::Size(),0);
  }

  //    std::vector<CPosDataset> tempPosSet(0);
  int currentClass = treeNum % tempClassDatabase.vNode.size();

  //std::cout << "okashiina" << std::endl;
  if(!conf.modelLearningMode)
    for(int i = 0; i < posSet.size(); ++i){
      if(tempClassDatabase.search(posSet.at(i).getClassName()) == currentClass){
	tempPosSet.push_back(posSet.at(i));
	//std::cout << "teketeke" << std::endl;
      }else{
	negSet.push_back(convertPosToNeg2(posSet.at(i)));
	//std::cout << "negneg" << std::endl;
      }
    }

  //posSet = tempPosSet;

  loadTrainNegFile(conf, negSet);

  std::cout << "dataset loaded" << std::endl;

  // initialize class database
  //classDatabase.clear();

  std::cout << "generating appearance from 3D model!" << std::endl;
  // extract pos features and register classDatabase
  for(int i = 0; i < posSet.size(); ++i){
    //std::cout << i << std::endl;

    //std::cout << posSet.at(i).rgb << std::endl;
    //#pragma omp critical
    if(conf.modelLearningMode)
      posSet.at(i).loadImage(conf, posSet.at(i).getModelPath(), posSet.at(i).getParam());
    else
      posSet.at(i).loadImage(conf);
    //        if(imgload == -1 && conf.learningMode != 2){
    //            std::cout << "can't load image files" << std::endl;
    //            exit(-1);
    //        }

    posSet.at(i).extractFeatures(conf);

    //std::cout << posSet.at(i).img.at(1)->type() << std::endl;
    //std::cout << "detayo" << std::endl;

    //std::cout << posSet.size() << std::endl;

    classDatabase.add(posSet.at(i).getParam()->getClassName(),posSet.at(i).img.at(0)->size(),0);
    pBar(i,posSet.size(),50);
  }
  std::cout << std::endl;
  classDatabase.show();

  // extract neg features
  for(int i = 0; i < negSet.size(); ++i){
    //        if(negSet.at(i).getModel() != NULL)
    //            negSet.at(i).loadImage(conf, negSet.at(i).getModelPath(), posSet.at(i))
    negSet.at(i).loadImage(conf);
    negSet.at(i).extractFeatures(conf);
  }

  CRTree *tree = new CRTree(conf.min_sample, conf.max_depth, classDatabase.vNode.size(),this->classDatabase);
  std::cout << "tree created" << std::endl;

  extractPosPatches(posSet,posPatch,conf,treeNum,this->classDatabase);
  extractNegPatches(negSet,negPatch,conf);

  std::cout << "extracted pathes" << std::endl;
  std::vector<int> patchClassNum(classDatabase.vNode.size(), 0);

  for(int j = 0; j < posPatch.size(); ++j)
    patchClassNum.at(classDatabase.search(posPatch.at(j).getClassName()))++;

  // grow tree
  //vTrees.at(treeNum)->growTree(vPatches, 0,0, (float)(vPatches.at(0).size()) / ((float)(vPatches.at(0).size()) + (float)(vPatches.at(1).size())), conf, gen, patchClassNum);
  //#pragma omp parallel
  tree->growTree(posPatch,negPatch, 0,0, ((float)posPatch.size() / (float)(posPatch.size() + negPatch.size())), conf, patchClassNum);

  //    cv::namedWindow("test");
  //    cv::imshow("test", *posSet.at(0).feature.at(3));
  //    cv::waitKey(0);
  //    cv::destroyAllWindows();

  // save tree
  sprintf(buffer, "%s%03d.txt",
	  conf.treepath.c_str(), treeNum + conf.off_tree);
  std::cout << "tree file name is " << buffer << std::endl;
  tree->saveTree(buffer);

  // save class database
  sprintf(buffer, "%s%s%03d.txt",
	  conf.treepath.c_str(),
	  conf.classDatabaseName.c_str(), treeNum + conf.off_tree);
  std::cout << "write tree data" << std::endl;
  classDatabase.write(buffer);

  //double time = t.elapsed();

  //std::cout << "tree " << treeNum << " calicuration time is " << time << std::endl;

  sprintf(buffer, "%s%03d_timeResult.txt",conf.treepath.c_str(), treeNum + conf.off_tree);
  std::fstream lerningResult(buffer, std::ios::out);
  if(lerningResult.fail()){
    std::cout << "can't write result" << std::endl;
  }

  lerningResult << time << std::endl;

  lerningResult.close();

  delete tree;


  posPatch.clear();
  negPatch.clear();

  //    for(int i = 0; i< posSet.size(); ++i){
  //        Cv::Namedwindow("Test");
  //        cv::imshow("test", *posSet.at(i).img.at(0));
  //        cv::waitKey(0);
  //        cv::destroyWindow("test");
  //    }


  //    for(int i = 0; i < posSet.size(); ++i){
  //        posSet.at(i).releaseImage();
  //    }

  //    for(int i = 0; i< posSet.size(); ++i){
  //        cv::namedWindow("test");
  //        cv::imshow("test", *posSet.at(i).feature.at(0));
  //        cv::waitKey(0);
  //        cv::destroyWindow("test");
  //    }

  //    for(int i = 0; i < posSet.size(); ++i){
  //        posSet.at(i).releaseFeatures();
  //    }

  posSet.clear();
  negSet.clear();
}

void CRForest::loadForest(){
  char buffer[256];
  char buffer2[256];
  std::cout << "loading forest..." << std::endl;
  for(int i = 0; i < vTrees.size(); ++i){
    sprintf(buffer, "%s%03d.txt",conf.treepath.c_str(),i);
    sprintf(buffer2, "%s%s%03d.txt", conf.treepath.c_str(), conf.classDatabaseName.c_str(), i);
    vTrees[i] = new CRTree(buffer, buffer2, conf);

    //std::cout << buffer2 << std::endl;
    classDatabase.read(buffer2);
    pBar(i,vTrees.size(),50);
  }
  std::cout << std::endl;
}

// name   : detect function
// input  : image and dataset
// output : classification result and detect picture
CDetectionResult CRForest::detection(CTestDataset &testSet) const{
  int classNum = classDatabase.vNode.size();//contain class number
  std::vector<CTestPatch> testPatch;
  std::vector<const LeafNode*> result;

  //std::vector<const LeafNode*> storedLN(0);
  //std::vector<std::vector<CParamset> > cluster(0);
  //std::vector<CParamset> clusterMean(0);

  cv::vector<cv::Mat> outputImage(classNum);
  cv::vector<cv::Mat> voteImage(classNum);//voteImage(classNum);
  //cv::vector<cv::Mat_<std::vector<CParamset> > > voteParam(classNum);

  std::vector<int> totalVote(classNum,0);

  //boost::timer t;

  //boost::timer::auto_cpu_timer t;
  //boost::timer::nanosecond_type time;

  std::vector<paramHist**> voteParam2(classNum);

  //timer.start();

  testSet.loadImage(conf);

  testSet.extractFeatures(conf);

  //std::cout << "extracted feature " << t.elapsed() << " sec" << std::endl;
    
  //testSet.releaseImage();

  //t.restart()
  // image row and col
  int imgRow = testSet.img.at(0)->rows;
  int imgCol = testSet.img.at(0)->cols;

#pragma omp parallel
  {
#pragma omp for
    for(int i = 0; i < classNum; ++i){
      outputImage.at(i) = testSet.img.at(0)->clone();
      voteImage.at(i) = cv::Mat::zeros(imgRow,imgCol,CV_32FC1);
      voteParam2.at(i) = new paramHist*[imgRow];
      for(int j = 0; j < imgRow; ++j)
	voteParam2.at(i)[j] = new paramHist[imgCol];
    }
  }

  //paramHist voteParam[testSet.img.at(0)->rows][testSet.img.at(0)->cols][classNum];
  // extract feature from test image
  //features.clear();
  //extractFeatureChannels(image.at(0), features);
  // add depth image to features
  //features.push_back(image.at(1));
  // extract patches from features

  extractTestPatches(testSet,testPatch,this->conf);

  //std::cout << "extracted feature " << t.elapsed() << " sec" << std::endl;

  std::cout << "patch num: " << testPatch.size() << std::endl;
  std::cout << "detecting..." << std::endl;
  // regression and vote for every patch

  std::cout << "class num = " << classNum << std::endl;

  for(int j = 0; j < testPatch.size(); ++j){
    // regression current patch
    result.clear();
    this->regression(result, testPatch.at(j));

    // for each tree leaf
    for(int m = 0; m < result.size(); ++m){
#pragma omp parallel
      {
#pragma omp for

	for(int l = 0; l < result.at(m)->pfg.size(); ++l){
	  if(result.at(m)->pfg.at(l) > 0.9){
	    int cl = classDatabase.search(result.at(m)->param.at(l).at(0).getClassName());

	    for(int n = 0; n < result.at(m)->param.at(cl).size(); ++n){
	      cv::Point patchSize(conf.p_height/2,conf.p_width/2);

	      cv::Point rPoint = result.at(m)->param.at(cl).at(n).getCenterPoint();

	      if(conf.learningMode != 2){
		cv::Mat tempDepth = *testPatch[j].getDepth();
		cv::Rect tempRect = testPatch[j].getRoi();
		cv::Mat realDepth = tempDepth(tempRect);
		double centerDepth = realDepth.at<ushort>(tempRect.height / 2 + 1, tempRect.width / 2 + 1) + conf.mindist;

		rPoint *= centerDepth;
		rPoint.x /= 1000;
		rPoint.y /= 1000;


	      }

	      cv::Point pos(testPatch.at(j).getRoi().x + patchSize.x +  rPoint.x,
			    testPatch.at(j).getRoi().y + patchSize.y +  rPoint.y);
	      // vote to result image
	      if(pos.x > 0 && pos.y > 0 && pos.x < voteImage.at(cl).cols && pos.y < voteImage.at(cl).rows){
		double v = result.at(m)->pfg.at(cl) / ( result.size() * result.at(m)->param.at(l).size());
		voteImage.at(cl).at<float>(pos.y,pos.x) += v;//(result.at(m)->pfg.at(c) - 0.9);// * 100;//weight * 500;
		voteParam2.at(cl)[pos.y][pos.x].roll.at<double>(0,result.at(m)->param.at(l).at(n).getAngle()[0]) += v * 10000;
		voteParam2.at(cl)[pos.y][pos.x].pitch.at<double>(0,result.at(m)->param.at(l).at(n).getAngle()[1]) += v * 10000;
		voteParam2.at(cl)[pos.y][pos.x].yaw.at<double>(0,result.at(m)->param.at(l).at(n).getAngle()[2]) += v * 10000;
		//std::cout << result.at(m)->param.at(l).at(n).getAngle() << std::endl;
		//std::cout << v << std::endl;
		totalVote.at(cl) += 1;
	      }

	    } //for(int n = 0; n < result.at(m)->param.at(cl).size(); ++n){
	  } //if(result.at(m)->pfg.at(l) > 0.9){
	} //for(int l = 0; l < result.at(m)->pfg.size(); ++l){

      }//pragma omp parallel

    } // for every leaf
  } // for every patch

    // vote end

#pragma omp parallel
  {
#pragma omp for
    // find balance by mean shift
    for(int i = 0; i < classNum; ++i){
      cv::GaussianBlur(voteImage.at(i),voteImage.at(i), cv::Size(21,21),0);
    }
  }

  // measure time
  //    double time = t.elapsed();
  //    std::cout << time << "sec" << std::endl;
  //    std::cout << 1 / (time) << "Hz" << std::endl;

  // output image to file
  std::string opath;
  //    if(!conf.demoMode){
  //        //create result directory
  //        opath = testSet.getRgbImagePath();
  //        opath.erase(opath.find_last_of(PATH_SEP));
  //        std::string imageFilename = testSet.getRgbImagePath();
  //        imageFilename.erase(imageFilename.find_last_of("."));
  //        //imageFilename.erase(imageFilename.begin(),imageFilename.find_last_of(PATH_SEP));
  //        imageFilename = imageFilename.substr(imageFilename.rfind(PATH_SEP),imageFilename.length());

  //        //opath += PATH_SEP;
  //        opath += imageFilename;
  //        std::string execstr = "mkdir -p ";
  //        execstr += opath;
  //        system( execstr.c_str() );

  //        for(int c = 0; c < classNum; ++c){
  //            std::stringstream cToString;
  //            cToString << c;
  //            std::string outputName = "output" + cToString.str() + ".png";
  //            std::string outputName2 = opath + PATH_SEP + "vote_" + classDatabase.vNode.at(c).name + ".png";
  //            //cv::imwrite(outputName.c_str(),outputImage.at(c));
  //            //cv::cvtColor(voteImage)

  //            cv::Mat writeImage;
  //            //hist.convertTo(hist, hist.type(), 200 * 1.0/second_val,0);
  //            voteImage.at(c).convertTo(writeImage, CV_8UC1, 254);
  //            cv::imwrite(outputName2.c_str(),writeImage);
  //        }
  //    }

  // create detection result
  CDetectionResult detectResult;
  detectResult.voteImage = voteImage;

  // show ground truth
  std::cout << "show ground truth" << std::endl;
  //    std::cout << dataSet.className.size() << std::endl;
  //    std::cout << dataSet.centerPoint.size() << std::endl;
  for(int i = 0; i < testSet.param.size(); ++i){
    testSet.param.at(i).showParam();
  }

  // show detection reslut
  std::cout << "show result" << std::endl;
  // for every class
  for(int c = 0; c < classNum; ++c){
    double min,max;
    cv::Point minLoc,maxLoc;
    cv::minMaxLoc(voteImage.at(c),&min,&max,&minLoc,&maxLoc);

    double min_pose_value[3], max_pose_value[3];
    cv::Point min_pose[3], max_pose[3];

    paramHist hist;

    for(int x = 0; x < conf.paramRadius; ++x){
      for(int y = 0; y < conf.paramRadius; ++y){
	if( maxLoc.x + x < imgCol &&  maxLoc.y + y < imgRow)
	  hist += voteParam2.at(c)[maxLoc.y + y][maxLoc.x + x];
	if(maxLoc.x - x > 0 && maxLoc.y - y > 0)
	  hist += voteParam2.at(c)[maxLoc.y - y][maxLoc.x - x];
      }
    }

    //hist.showHist();

    //        for(int p = 0; p < 360; ++p){
    //            std::cout << p << " " <<  voteParam2.at(c)[maxLoc.y][maxLoc.x].yaw.at<double>(0,p) << std::endl;
    //        }

    //voteParam2.at(c)[maxLoc.y][maxLoc.x].showHist();

    cv::minMaxLoc(hist.roll, &min_pose_value[0], &max_pose_value[0], &min_pose[0], &max_pose[0]);
    cv::minMaxLoc(hist.pitch, &min_pose_value[1], &max_pose_value[1], &min_pose[1], &max_pose[1]);
    cv::minMaxLoc(hist.yaw, &min_pose_value[2], &max_pose_value[2], &min_pose[2], &max_pose[2]);

    // draw detected class bounding box to result image
    // if you whant add condition of detection threshold, add here
    cv::Size tempSize = classDatabase.vNode.at(c).classSize;
    cv::Rect_<int> outRect(maxLoc.x - tempSize.width / 2,maxLoc.y - tempSize.height / 2 , tempSize.width,tempSize.height);
    cv::rectangle(outputImage.at(c),outRect,cv::Scalar(0,0,200),3);
    cv::putText(outputImage.at(c),classDatabase.vNode.at(c).name,cv::Point(outRect.x,outRect.y),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(0,0,200), 2, CV_AA);

    // draw grand truth to result image
    if(!conf.demoMode){
      for(int i = 0; i < testSet.param.size(); ++i){
	int tempClassNum = classDatabase.search(testSet.param.at(i).getClassName());
	if(tempClassNum != -1){
	  cv::Size tempSize = classDatabase.vNode.at(tempClassNum).classSize;
	  cv::Rect_<int> outRect(testSet.param.at(i).getCenterPoint().x - tempSize.width / 2,testSet.param.at(i).getCenterPoint().y - tempSize.height / 2 , tempSize.width,tempSize.height);
	  cv::rectangle(outputImage.at(tempClassNum),outRect,cv::Scalar(200,0,0),3);
	  cv::putText(outputImage.at(tempClassNum),classDatabase.vNode.at(c).name,cv::Point(testSet.param.at(i).getCenterPoint().x, testSet.param.at(i).getCenterPoint().y),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(200,0,0), 2, CV_AA);
	}
      }
    }

    // show result
    std::cout << c << " Name : " << classDatabase.vNode.at(c).name <<
      "\tvote : " << totalVote.at(c) <<
      " Score : " << voteImage.at(c).at<float>(maxLoc.y, maxLoc.x) <<
      " CenterPoint : " << maxLoc << std::endl <<
      " Pose : roll " << max_pose[0].x <<
      " pitch : " << max_pose[1].x <<
      " yaw : " << max_pose[2].x << std::endl;

    // if not in demo mode, output image to file
    if(!conf.demoMode){
      std::string outputName = opath + PATH_SEP + "detectionResult" + "_" + classDatabase.vNode.at(c).name + ".png";
      cv::imwrite(outputName.c_str(),outputImage.at(c));
    }

    CDetectedClass detectedClass;
    detectedClass.name = classDatabase.vNode.at(c).name;
    detectedClass.angle[0] = max_pose[0].x;

    // calc euclidean dist to nearest object
    double minError = DBL_MAX;
    std::string nearestObject;
    for(int d = 0; d < testSet.param.size(); ++d){
      double tempError = euclideanDist(maxLoc,testSet.param.at(d).getCenterPoint());//= std::sqrt(std::pow((double)(maxLoc.x - testSet.param.at(0).getCenterPoint().x), 2) + std::pow((double)(maxLoc.y - testSet.param.at(0).getCenterPoint().y), 2));
      //std::cout << tempError << std::endl;
      if(tempError < minError){
	minError = tempError;
	nearestObject = testSet.param.at(d).getClassName();
      }
    }

    // calc and output result
    detectedClass.error = minError;
    detectedClass.nearestClass = nearestObject;
    detectedClass.score = voteImage.at(c).at<float>(maxLoc.y, maxLoc.x);
    detectResult.detectedClass.push_back(detectedClass);
  } // for every class

  for(int k = 0; k < classNum; ++k){
    for(int i = 0; i < imgRow; ++i){
      delete[] voteParam2.at(k)[i];
    }
  }

  return detectResult;
}

// Regression
void CRForest::regression(std::vector<const LeafNode*>& result, CTestPatch &patch) const{
  result.resize( vTrees.size() );
  //std::cout << "enter regression" << std::endl;
  for(int i=0; i < vTrees.size(); ++i) {
    //std::cout << "regressioning " << i << std::endl;
    result[i] = vTrees[i]->regression(patch);
  }
}


