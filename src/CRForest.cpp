#include <boost/timer.hpp>
#include "CRForest.h"

// paramBin& paramBin::operator +(const paramBin& obj){ 
//   this->roll += obj.roll;
//   this->pitch += obj.pitch;
//   this->yaw += obj.yaw;

//   return *this;
// }

// paramBin& paramBin::operator +=(const paramBin& obj){
//   this->roll += obj.roll;
//   this->pitch += obj.pitch;
//   this->yaw += obj.yaw;

//   return *this;
// }

cv::Mat calcGaussian(double score, double center){
  cv::Mat vote = cv::Mat::zeros(1, 720, CV_32FC1);
  for(int i = -30; i <= 30; ++i){
    vote.at<float>(0, center + i + 180.0) += score * exp( -1 * abs( i * i ) / 2.0 );

    //    std::cout << i << " " << exp( -1 * abs( i * i ) / 2.0 ) << std::endl;
  }
  //  std::cout << std::endl;
  //int o;
  //std::cin >> o;
  return vote;
}

double euclideanDist(cv::Point_<double> p, cv::Point_<double> q)
{
  cv::Point_<double> diff = p - q;
  return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

void CRForest::learning(){

#pragma omp parallel
  {
#pragma omp for
    for(int i = 0;i < conf.ntrees; ++i){
      growATree(i);
    } // end tree loop
  }
}

void CRForest::growATree(const int treeNum){
  // positive, negative dataset
  std::vector<CPosDataset*> posSet(0);
  std::vector<CNegDataset*> negSet(0);

  // positive, negative patch
  std::vector<CPosPatch> posPatch(0);
  std::vector<CNegPatch> negPatch(0);

  char buffer[256];

  std::cout << "tree number " << treeNum << std::endl;

  // initialize random seed
  boost::mt19937    gen( treeNum * static_cast<unsigned long>(time(NULL)) );
  //boost::timer t;


  //loadTrainPosFile(conf, posSet);//, gen);
  //  if(conf.modelLearningMode)
  //  loadTrainObjFile(conf,posSet);
  //else
  loadTrainPosFile(conf, posSet);

  CClassDatabase tempClassDatabase;
  // extract pos features and register classDatabase
  for(unsigned int i = 0; i < posSet.size(); ++i){
    tempClassDatabase.add(posSet[i]->getParam()->getClassName(),cv::Size(),0);
  }


  loadTrainNegFile(conf, negSet);

  std::cout << "dataset loaded" << std::endl;

  std::cout << "generating appearance from 3D model!" << std::endl;
  // extract pos features and register classDatabase
  for(unsigned int i = 0; i < posSet.size(); ++i){
    posSet.at(i)->loadImage(conf);
    posSet.at(i)->extractFeatures(conf);
    classDatabase.add(posSet.at(i)->getParam()->getClassName(),posSet.at(i)->img.at(0)->size(),0);
    pBar(i,posSet.size(),50);
  }
  std::cout << std::endl;
  classDatabase.show();

  // extract neg features
  for(unsigned int i = 0; i < negSet.size(); ++i){
    negSet.at(i)->loadImage(conf);
    negSet.at(i)->extractFeatures(conf);
  }

  std::vector<CPosDataset*> tempPosSet(0);
  int currentClass = treeNum % tempClassDatabase.vNode.size();

  for(unsigned int i = 0; i < posSet.size(); ++i){
    if(tempClassDatabase.search(posSet.at(i)->getClassName()) == currentClass){
      tempPosSet.push_back(posSet.at(i));
    }else{
      negSet.push_back((CNegDataset*)posSet[i]);
    }
  }

  posSet = tempPosSet;

  CRTree *tree = new CRTree(conf.min_sample, conf.max_depth, classDatabase.vNode.size(),this->classDatabase);
  std::cout << "tree created" << std::endl;

  extractPosPatches(posSet,posPatch,conf,treeNum,this->classDatabase);
  extractNegPatches(negSet,negPatch,conf);

  std::cout << "extracted pathes" << std::endl;
  std::vector<int> patchClassNum(classDatabase.vNode.size(), 0);

  for(unsigned int j = 0; j < posPatch.size(); ++j)
    patchClassNum.at(classDatabase.search(posPatch.at(j).getClassName()))++;

  tree->growTree(posPatch,negPatch, 0,0, ((float)posPatch.size() / (float)(posPatch.size() + negPatch.size())), conf, patchClassNum);

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

  sprintf(buffer, "%s%03d_timeResult.txt",conf.treepath.c_str(), treeNum + conf.off_tree);
  std::fstream lerningResult(buffer, std::ios::out);
  if(lerningResult.fail()){
    std::cout << "can't write result" << std::endl;
  }

  lerningResult.close();

  delete tree;

  posPatch.clear();
  negPatch.clear();

  for(unsigned int i = 0; i < posSet.size(); ++i)
    delete posSet[i];
  for(unsigned int i = 0; i < negSet.size(); ++i)
    delete negSet[i];

  posSet.clear();
  std::cout << negSet.size() << std::endl;
  negSet.clear();
}

void CRForest::loadForest(){
  char buffer[256];
  char buffer2[256];
  std::cout << "loading forest..." << std::endl;
  for(unsigned int i = 0; i < vTrees.size(); ++i){
    sprintf(buffer, "%s%03d.txt",conf.treepath.c_str(),i);
    sprintf(buffer2, "%s%s%03d.txt", conf.treepath.c_str(), conf.classDatabaseName.c_str(), i);
    vTrees[i] = new CRTree(buffer, buffer2, conf);

    classDatabase.read(buffer2);
    pBar(i,vTrees.size(),50);
  }
  std::cout << std::endl;
}

// name   : detect function
// input  : image and dataset
// output : classification result and detect picture
CDetectionResult CRForest::detection(CTestDataset &testSet) const{
  std::cout << "detection started" << std::endl;

  int classNum = classDatabase.vNode.size(); //contain class number
  std::vector<CTestPatch> testPatch;
  std::vector<const LeafNode*> result;

  cv::vector<cv::Mat> outputImage(classNum);

  cv::vector<cv::Mat> voteImage(classNum);
  std::vector<int> classVoteNum(classNum,0);
  std::vector<boost::shared_ptr<paramBin>**> paramVote(classNum);

  // image row and col
  int imgRow = testSet.img.at(0)->rows;
  int imgCol = testSet.img.at(0)->cols;

  for(int i = 0; i < classNum; ++i){
    outputImage[i] = testSet.img[0]->clone();
    voteImage[i] = cv::Mat::zeros(imgRow,imgCol,CV_32FC1);
    paramVote[i] = new boost::shared_ptr<paramBin>*[imgRow];
    for(int j = 0; j < imgRow; ++j)
      paramVote[i][j] = new boost::shared_ptr<paramBin>[imgCol];
  }


  testSet.extractFeatures(conf);

  //std::cout << testSet.getRgbImagePath() << std::endl;

  std::cout << "feature extracted" << std::endl;

  cv::Mat votedVectors = cv::Mat::zeros(imgRow, imgCol, CV_8UC3);

  //std::cout << "vote image col " << (int)((double)imgCol / (double)conf.stride + 0.5) +1 << std::endl;
  //std::cout << "vote image row " << (int)((double)imgRow / (double)conf.stride + 0.5) +1 << std::endl;



  extractTestPatches(&testSet,testPatch,this->conf);

  std::cout << "patch num: " << testPatch.size() << std::endl;
  std::cout << "detecting..." << std::endl;
  std::cout << "class num = " << classNum << std::endl;

  for(unsigned int j = 0; j < testPatch.size(); ++j){
    // regression current patch
    result.clear();
    this->regression(result, testPatch.at(j));

    //std::cout << "regression end" << j << std::endl;

    // for each tree leaf
    for(unsigned int m = 0; m < result.size(); ++m){
#pragma omp parallel
      {
#pragma omp for
	for(unsigned int l = 0; l < result.at(m)->pfg.size(); ++l){
	  if(result.at(m)->pfg.at(l) > 0.9){
	    int cl = classDatabase.search(result.at(m)->param.at(l).at(0).getClassName());

	    for(unsigned int n = 0; n < result.at(m)->param.at(cl).size(); ++n){
	      cv::Point_<double> patchSize(conf.p_height/2,conf.p_width/2);

	      cv::Point_<double> rPoint = result.at(m)->param.at(cl).at(n).getCenterPoint();

	      if(conf.learningMode != 2){
		cv::Mat tempDepth = *testPatch[j].getDepth();
		cv::Rect tempRect = testPatch[j].getRoi();
		cv::Mat realDepth = tempDepth(tempRect);
		double centerDepth = realDepth.at<ushort>(tempRect.height / 2 + 1, tempRect.width / 2 + 1) + conf.mindist;



		rPoint.x *= centerDepth;
		rPoint.y *= centerDepth;
		rPoint.x /= 100;
		rPoint.y /= 100;

		//std::cout << rPoint << std::endl;		 
		// rPoint.x = 0;
		// rPoint.y = 0;


	      }

	      cv::Point pos(testPatch.at(j).getRoi().x +  rPoint.x,
			    testPatch.at(j).getRoi().y +  rPoint.y);

	      //std::cout << "voting" << std::endl;
	      // vote to result image
	      if(pos.x > 0 && pos.y > 0 && pos.x < voteImage.at(cl).cols && pos.y < voteImage.at(cl).rows){
		double v = result.at(m)->pfg.at(cl) / ( result.size() * result.at(m)->param.at(l).size()) / (euclideanDist(cv::Point(), rPoint) + 1);


		voteImage.at(cl).at<float>(pos.y,pos.x) += v * 1000000;//(result.at(m)->pfg.at(c) - 0.9);// * 100;//weight * 500;
		if(!conf.tsukubaMode){
		  double ta[3] = {result.at(m)->param.at(l).at(n).getAngle()[0],
				  result.at(m)->param.at(l).at(n).getAngle()[1],
				  result.at(m)->param.at(l).at(n).getAngle()[2]};

		  if(paramVote[cl][pos.y][pos.x])
		    paramVote[cl][pos.y][pos.x]->addChild(v, ta[0], ta[1], ta[2]);
		  else
		    paramVote[cl][pos.y][pos.x] 
		      = boost::shared_ptr<paramBin>
		      (new paramBin(v , ta[0], ta[1], ta[2]));
		}
		//if((int)((double)pos.y / (double)conf.stride) < (int)((double)imgRow / (double)conf.stride + 0.5) && (int)((double)pos.x / (double)conf.stride) < (int)((double)imgCol / (double)conf.stride + 0.5)){

		//paramVote.at(cl)[(int)((double)pos.y / (double)conf.stride)][(int)((double)pos.x / (double)conf.stride)].roll.at<double>(0,result.at(m)->param.at(l).at(n).getAngle()[0]) += v * 1000;
		//paramVote.at(cl)[(int)((double)pos.y / (double)conf.stride)][(int)((double)pos.x / (double)conf.stride)].pitch.at<double>(0,result.at(m)->param.at(l).at(n).getAngle()[1]) += v * 1000;
		//paramVote.at(cl)[(int)((double)pos.y / (double)conf.stride)][(int)((double)pos.x / (double)conf.stride)].yaw.at<double>(0,result.at(m)->param.at(l).at(n).getAngle()[2]) += v * 1000;
		//}

		cv::Scalar hanabi;

		switch(cl){
		case 0:
		  hanabi= cv::Scalar(0,0,255);
		  break;
		case 1:
		  hanabi= cv::Scalar(255,0,0);
		  break;
		case 2:
		  hanabi= cv::Scalar(0,255,0);
		  break;
		}
		// this code is for debug and setting
		cv::circle(votedVectors,pos,5,hanabi);//cv::Scalar(254,254,254));
		cv::line(votedVectors,pos,
			 cv::Point(testPatch[j].getRoi().x,testPatch[j].getRoi().y),
			 hanabi);///cv::Scalar(245,245,245));
		classVoteNum.at(cl) += 1;
	      }

	    } //for(int n = 0; n < result.at(m)->param.at(cl).size(); ++n){
	  } //if(result.at(m)->pfg.at(l) > 0.9){
	} //for(int l = 0; l < result.at(m)->pfg.size(); ++l){

      }//pragma omp parallel

    } // for every leaf
  } // for every patch

  // vote end
  std::cout <<  "vote end" << std::endl;

#pragma omp parallel
  {
#pragma omp for
    // find balance by mean shift
    for(int i = 0; i < classNum; ++i){
      cv::GaussianBlur(voteImage.at(i),voteImage.at(i), cv::Size(101,101),0);
    }
  }

  cv::imshow("vote", voteImage.at(2));
  cv::waitKey(1);

  // output image to file
  std::string opath;
  cv::Mat showVoteImage = cv::Mat(voteImage.at(0).rows, voteImage.at(0).cols, CV_8UC1);

  // create detection result
  CDetectionResult detectResult;
  detectResult.voteImage = voteImage;

  // show ground truth
  std::cout << "show ground truth" << std::endl;
  //    std::cout << dataSet.className.size() << std::endl;
  //    std::cout << dataSet.centerPoint.size() << std::endl;
  for(unsigned int i = 0; i < testSet.param.size(); ++i){
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

    cv::Mat voteAngle = cv::Mat::zeros(3, 720, CV_32FC1);
    //for(int i = 0; i < 3; ++i)
    //voteAngle[i] = cv::Mat::zeros(1, 640);

    //paramBin hist;

    if(!conf.tsukubaMode){
      for(int x = 0; x < 5; ++x){
	for(int y = 0; y < 5; ++y){
	  if(maxLoc.x + x < imgCol && maxLoc.y + y < imgRow){
	    boost::shared_ptr<paramBin> pBin = paramVote[c][maxLoc.y + y][maxLoc.x +x];
	    while(pBin){
	      voteAngle.row(0) += calcGaussian(pBin->confidence, pBin->roll);//at<>[0][pBin->roll] 
	      voteAngle.row(1) += calcGaussian(pBin->confidence, pBin->pitch);//at<>[0][pBin->roll] 
	      voteAngle.row(2) += calcGaussian(pBin->confidence, pBin->yaw);//at<>[0][pBin->roll] 
	      pBin = pBin->next;
	    }
	  }
	}
      }
    


      //std::cout << voteAngle << std::endl;

      // for(int x = 0; x < conf.paramRadius - 1; ++x){
      //   for(int y = 0; y < conf.paramRadius - 1; ++y){
      // 	if(maxLoc.x + x >= 0 && maxLoc.y + y >= 0 && maxLoc.x + x < imgCol - conf.stride &&  maxLoc.y + y < imgRow - conf.stride){
	  
      // 	  hist += paramVote.at(c)[(int)((double)(maxLoc.y + y) / (double)conf.stride)][(int)((double)(maxLoc.x + x) / (double)conf.stride)];


      // 	}

      // 	if(maxLoc.x - x >= 0 && maxLoc.y - y >= 0 && maxLoc.x - x < imgCol - conf.stride &&  maxLoc.y - y < imgRow - conf.stride){
	  
      // 	  hist += paramVote.at(c)[(int)((double)(maxLoc.y - y) / (double)conf.stride)][(int)((double)(maxLoc.x - x) / (double)conf.stride)];
	

      // 	}
      //   }
      // }

      cv::minMaxLoc(voteAngle.row(0), &min_pose_value[0], &max_pose_value[0], &min_pose[0], &max_pose[0]);
      cv::minMaxLoc(voteAngle.row(1), &min_pose_value[1], &max_pose_value[1], &min_pose[1], &max_pose[1]);
      cv::minMaxLoc(voteAngle.row(2), &min_pose_value[2], &max_pose_value[2], &min_pose[2], &max_pose[2]);

    }

    // draw detected class bounding box to result image
    // if you whant add condition of detection threshold, add here
    cv::Size tempSize = classDatabase.vNode.at(c).classSize;
    cv::Rect_<int> outRect(maxLoc.x - tempSize.width / 2,maxLoc.y - tempSize.height / 2 , tempSize.width,tempSize.height);
    cv::rectangle(outputImage.at(c),outRect,cv::Scalar(0,0,200),3);
    cv::putText(outputImage.at(c),classDatabase.vNode.at(c).name,cv::Point(outRect.x,outRect.y),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(0,0,200), 2, CV_AA);

    cv::circle(outputImage[0], maxLoc, 10, cv::Scalar(200,0,0));
    cv::putText(outputImage[0],classDatabase.vNode.at(c).name,maxLoc,cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(200,0,0), 2, CV_AA);

    // draw grand truth to result image
    if(!conf.demoMode){
      for(unsigned int i = 0; i < testSet.param.size(); ++i){
	int tempClassNum = classDatabase.search(testSet.param.at(i).getClassName());
	if(tempClassNum != -1){
	  cv::Size tempSize = classDatabase.vNode.at(tempClassNum).classSize;
	  cv::Rect_<int> outRect(testSet.param.at(i).getCenterPoint().x - tempSize.width / 2,testSet.param.at(i).getCenterPoint().y - tempSize.height / 2 , tempSize.width,tempSize.height);
	  //cv::rectangle(outputImage[0],outRect,cv::Scalar(200,0,0),3);
	  cv::circle(outputImage[0], maxLoc, 20, cv::Scalar(200,0,0));
	  cv::putText(outputImage[0],classDatabase.vNode.at(c).name,maxLoc,cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(200,0,0), 2, CV_AA);
	}
      }
    }



    // show result
    std::cout << c << " Name : " << classDatabase.vNode.at(c).name <<
      "\tvote : " << classVoteNum.at(c) <<
      " Score : " << voteImage.at(c).at<float>(maxLoc.y, maxLoc.x) <<
      " CenterPoint : " << maxLoc << std::endl <<
      " Pose : roll " << max_pose[0].x - 180 <<
      " pitch : " << max_pose[1].x - 180 <<
      " yaw : " << max_pose[2].x - 180 << std::endl;

    

    // if not in demo mode, output image to file
    if(!conf.demoMode){
      std::string outputName = opath + PATH_SEP + "detectionResult" + "_" + classDatabase.vNode.at(c).name + ".png";
      cv::imwrite(outputName.c_str(),outputImage.at(c));
    }

    

    CDetectedClass detectedClass;
    detectedClass.name = classDatabase.vNode.at(c).name;
    detectedClass.angle[0] = max_pose[0].x;

    double minError = 10000000;

    std::string nearestObject;

    for(unsigned int d = 0; d < testSet.param.size(); ++d){
      double tempError = euclideanDist(maxLoc,testSet.param.at(d).getCenterPoint());std::sqrt(std::pow((double)(maxLoc.x - testSet.param.at(0).getCenterPoint().x), 2) + std::pow((double)(maxLoc.y - testSet.param.at(0).getCenterPoint().y), 2));
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
    detectedClass.centerPoint = maxLoc;
    detectResult.detectedClass.push_back(detectedClass);
  } // for every class

    // if(!conf.demoMode){
    //   cv::namedWindow("test");
    //   cv::imshow("test", showVoteImage);
    //   cv::namedWindow("test2");
    //   cv::imshow("test2", outputImage[0]);
    //   cv::namedWindow("test3");
    //   cv::imshow("test3", votedVectors);
    //   cv::imwrite("hanabi.png",votedVectors);
  
    //   cv::waitKey(0);
    // }

  for(int k = 0; k < classNum; ++k){
    for(int i = 0; i < imgRow; ++i){
      delete[] paramVote[k][i];
    }
  }

  return detectResult;
}

// Regression
void CRForest::regression(std::vector<const LeafNode*>& result, CTestPatch &patch) const{
  result.resize( vTrees.size() );
  //std::cout << "enter regression" << std::endl;
  for(unsigned int i=0; i < vTrees.size(); ++i) {
    //std::cout << "regressioning " << i << std::endl;
    result[i] = vTrees[i]->regression(patch);
  }
}

