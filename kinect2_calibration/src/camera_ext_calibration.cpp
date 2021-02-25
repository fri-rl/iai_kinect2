/**
 * Copyright 2014 University of Bremen, Institute for Artificial Intelligence
 * Author: Thiemo Wiedemeyer <wiedemeyer@cs.uni-bremen.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <regex>
#include <vector>
#include <mutex>
#include <thread>

#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>

// If OpenCV4
#if CV_VERSION_MAJOR > 3
#include <opencv2/imgcodecs/legacy/constants_c.h>
#endif

#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/TransformStamped.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

// #include <message_filters/subscriber.h>
// #include <message_filters/synchronizer.h>
// #include <message_filters/sync_policies/approximate_time.h>

#include <kinect2_calibration/kinect2_calibration_definitions.h>
#include <kinect2_bridge/kinect2_definitions.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Transform.h>
// #include <tf2/tf2_geometry_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

void tf2_to_opencv(const tf2::Transform &transform, cv::Vec3d &cv_translation, cv::Vec3d &cv_rotation){

  tf2::Quaternion quaternion = transform.getRotation();
  tf2::Vector3 tf2_translation = transform.getOrigin();
  double tf2_angle = quaternion.getAngle();
  tf2::Vector3 tf2_axis = quaternion.getAxis();

  cv_translation[0] = tf2_translation[0];
  cv_translation[1] = tf2_translation[1];
  cv_translation[2] = tf2_translation[2];

  cv_rotation[0] = tf2_axis[0]*tf2_angle;
  cv_rotation[1] = tf2_axis[1]*tf2_angle;
  cv_rotation[2] = tf2_axis[2]*tf2_angle;
}

void opencv_to_tf2(const cv::Vec3d &cv_translation, const cv::Vec3d &cv_rotation, tf2::Transform &transform){

  transform.setOrigin(tf2::Vector3(cv_translation[0], cv_translation[1], cv_translation[2]));
  double cv_angle = sqrt(pow(cv_rotation[0],2.0)+pow(cv_rotation[1],2.0)+pow(cv_rotation[2],2.0));
  if (cv_angle == 0.0){
    transform.setRotation(tf2::Quaternion(0.0,0.0,0.0,1.0));
  }else{
    transform.setRotation(tf2::Quaternion(tf2::Vector3(cv_rotation[0]/cv_angle, cv_rotation[1]/cv_angle, cv_rotation[2]/cv_angle), cv_angle));
  }

}



enum Board
{
  CHESS,
  CIRCLE,
  ACIRCLE,
  CHARUCO
};


class CameraCalibration
{
private:

  const ros::NodeHandle nh;
  const std::string path;

  cv::Size imageSize;
  const Board board_type;
  const cv::Size boardDims;
  const float boardSize;


  // std::vector<cv::Point3f> board;

  std::vector<std::vector<cv::Point2f> > pointsA;
  std::vector<std::vector<cv::Point2f> > pointsB;
  std::vector<std::vector<int> > idsA;
  std::vector<std::vector<int> > idsB;
  std::vector<std::string> imagesA;
  std::vector<std::string> imagesB;

  std::vector<cv::Vec3d> refA_trans, refA_rot, refB_trans, refB_rot;

  cv::Mat cameraMatrixA, distortionA;
  cv::Mat cameraMatrixB, distortionB;
  cv::Mat essential, fundamental, disparity;
  // cv::Mat rotation, translation;

  std::vector<cv::Mat> rvecs, tvecs;

  cv::Ptr<cv::aruco::Dictionary> aruco_dictionary;
  cv::Ptr<cv::aruco::CharucoBoard> charuco_board;
  cv::Ptr<cv::aruco::Board> aruco_board;


public:



  CameraCalibration(const ros::NodeHandle &nh, const std::string &path, const cv::Size imageSize, const Board board_type, const cv::Size &boardDims, const float boardSize, const int aruco_dict_id)
      : nh(nh),  path(path), imageSize(imageSize), board_type(board_type), boardDims(boardDims), boardSize(boardSize)
  {
    // board.resize(boardDims.width * boardDims.height);

    //##################################################################
    //############# Commented out since currently not supported ########
    //##################################################################
    //
    // if (board_type == ACIRCLE)
    // {
    //   for (size_t r = 0, i = 0; r < (size_t)boardDims.height; ++r)
    //   {
    //     for (size_t c = 0; c < (size_t)boardDims.width; ++c, ++i)
    //     {
    //       board[i] = cv::Point3f(float((2 * c + r % 2) * boardSize), float(r * boardSize), 0); //for asymmetrical circles
    //     }
    //   }
    // }
    // else
    // {
    //   for (size_t r = 0, i = 0; r < (size_t)boardDims.height; ++r)
    //   {
    //     for (size_t c = 0; c < (size_t)boardDims.width; ++c, ++i)
    //     {
    //       board[i] = cv::Point3f(c * boardSize, r * boardSize, 0);
    //     }
    //   }
    // }

    if (board_type == CHARUCO){
      aruco_dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(cv::aruco::DICT_6X6_250));
      // create charuco board object
      // charuco_board = cv::aruco::CharucoBoard::create(boardDims.width, boardDims.height, boardSize, 0.05, aruco_dictionary);
      charuco_board = cv::aruco::CharucoBoard::create(5, 6, 0.14, 0.10, aruco_dictionary);
      aruco_board = charuco_board.staticCast<cv::aruco::Board>();
    }

  }

  ~CameraCalibration()
  {
  }

  bool restore()
  {
    std::vector<std::string> frames;


    DIR *dp;
    struct dirent *dirp;
    size_t posA, frameNr_end;

    if((dp  = opendir(path.c_str())) ==  NULL){
      OUT_ERROR("Error opening: " << path);
      return false;
    }

    while((dirp = readdir(dp)) != NULL){
      std::string filename = dirp->d_name;

      if(dirp->d_type != DT_REG)
      {
        continue;
      }

      posA = filename.rfind("_ext_A.png");

      if(posA != std::string::npos)
      {
        frameNr_end = filename.find("_");
        if (frameNr_end != std::string::npos) {
          std::string frameName = filename.substr(0, frameNr_end);
          frames.push_back(frameName);
        }
      }

    }
    closedir(dp);

    std::sort(frames.begin(), frames.end());

    bool ret = true;
    if(frames.empty())
    {
      OUT_ERROR("no files found!");
      return false;
    }


    pointsA.resize(frames.size());
    imagesA.resize(frames.size());
    idsA.resize(frames.size());
    pointsB.resize(frames.size());
    imagesB.resize(frames.size());
    idsB.resize(frames.size());
    refA_trans.resize(frames.size());
    refB_trans.resize(frames.size());
    refA_rot.resize(frames.size());
    refB_rot.resize(frames.size());
    ret = ret && readFiles(frames, pointsA, pointsB, idsA, idsB, imagesA, imagesB, refA_trans, refA_rot, refB_trans, refB_rot);
    ret = ret && checkSyncPointsOrder();
    ret = ret && loadCalibration();

    return ret;
  }



  void calibrate(cv::Vec3d &translation, cv::Vec3d &rotation)
  {
    if (board_type == CHARUCO){
      calibrateSyncCharuco(translation, rotation);
    }else{
      OUT_ERROR("currently not supported!");
      //calibrateSync();
    }
    storeCalibration(translation, rotation);
  }

private:



  bool readFiles(const std::vector<std::string> &frames, std::vector<std::vector<cv::Point2f> > &pointsA, std::vector<std::vector<cv::Point2f> > &pointsB, std::vector<std::vector<int> > &idsA, std::vector<std::vector<int> > &idsB, std::vector<std::string> &imagesA, std::vector<std::string> &imagesB, std::vector<cv::Vec3d> &refA_trans, std::vector<cv::Vec3d> &refA_rot, std::vector<cv::Vec3d> &refB_trans, std::vector<cv::Vec3d> &refB_rot  ) const
  {
    bool ret = true;

    #pragma omp parallel for
    for(size_t i = 0; i < frames.size(); ++i)
    {
      std::string pointsname = path + "/" + frames[i] + "_ext.yaml";

      #pragma omp critical
      OUT_INFO("restoring frame: " << frames[i] << " from " << pointsname);

      cv::FileStorage file(pointsname, cv::FileStorage::READ);
      if(!file.isOpened())
      {
        #pragma omp critical
        {
          ret = false;
          OUT_ERROR("couldn't open pointsfile for frame: " << frames[i]);
        }
      }
      else
      {
        file["pointsA"] >> pointsA[i];
        file["idsA"] >> idsA[i];
        file["pointsB"] >> pointsB[i];
        file["idsB"] >> idsB[i];
        file["refA_trans"] >> refA_trans[i];
        file["refA_rot"] >> refA_rot[i];
        file["refB_trans"] >> refB_trans[i];
        file["refB_rot"] >> refB_rot[i];
        imagesA[i] = path +"/"+ frames[i] + "_ext_A.png";
        imagesB[i] = path +"/"+ frames[i] + "_ext_B.png";
      }
    }

    // for(size_t i = 1; i < frames.size(); ++i){
    //   if (cv::norm(refA_trans[0],refA_trans[i]) > 0.001){
    //     ret = false;
    //     OUT_ERROR("Difference in translation of reference frames for A too large " << cv::norm(refA_trans[0],refA_trans[i]) << ".\nCurrently only static camera positions are supported");
    //   }
    //   if (cv::norm(refA_rot[0],refA_rot[i]) > 0.001){
    //     ret = false;
    //     OUT_ERROR("Difference in rotation of reference frames for A too large " << cv::norm(refA_rot[0],refA_rot[i]) << ".\nCurrently only static camera positions are supported");
    //   }
    //   if (cv::norm(refB_trans[0],refB_trans[i]) > 0.001){
    //     ret = false;
    //     OUT_ERROR("Difference in translation of reference frames for B too large " << cv::norm(refB_trans[0],refB_trans[i]) << ".\nCurrently only static camera positions are supported");
    //   }
    //   if (cv::norm(refB_rot[0],refB_rot[i]) > 0.001){
    //     ret = false;
    //     OUT_ERROR("Difference in rotation of reference frames for B too large " << cv::norm(refB_rot[0],refB_rot[i]) << ".\nCurrently only static camera positions are supported");
    //   }
    // }

    return ret;
  }

  bool checkSyncPointsOrder()
  {
    if(pointsA.size() != pointsB.size())
    {
      OUT_ERROR("number of detected A and B patterns does not match!");
      return false;
    }

    if (board_type == CHARUCO){
      return true;
    }

    //##################################################################
    //############# Commented out since currently not supported ########
    //##################################################################
    //
    // for(size_t i = 0; i < pointsColor.size(); ++i)
    // {
    //   const std::vector<cv::Point2f> &pColor = pointsColor[i];
    //   const std::vector<cv::Point2f> &pIr = pointsIr[i];
    //
    //   if(pColor.front().y > pColor.back().y || pColor.front().x > pColor.back().x)
    //   {
    //     std::reverse(pointsColor[i].begin(), pointsColor[i].end());
    //   }
    //
    //   if(pIr.front().y > pIr.back().y || pIr.front().x > pIr.back().x)
    //   {
    //     std::reverse(pointsIr[i].begin(), pointsIr[i].end());
    //   }
    // }
    return false;
  }


    void calibrateSyncCharuco(cv::Vec3d &translation, cv::Vec3d &rotation)
    {
      if(pointsA.size() != pointsB.size())
      {
        OUT_ERROR("number of detected A and B patterns does not match!");
        return;
      }

      if(pointsA.empty() || pointsB.empty())
      {
        OUT_ERROR("no data for calibration provided!");
        return;
      }
      const cv::TermCriteria termCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);
      double error;

      // OUT_INFO("Camera Matrix A:" << std::endl << cameraMatrixA);
      // OUT_INFO("Distortion Coeeficients A:" << std::endl << distortionA << std::endl);
      // OUT_INFO("Camera Matrix B:" << std::endl << cameraMatrixB);
      // OUT_INFO("Distortion Coeeficients B:" << std::endl << distortionB << std::endl);

      OUT_INFO("Pairing and Filtering charuco points...");
      std::vector<std::vector<cv::Point3f> > _pointsBoard;
      std::vector<std::vector<cv::Point2f> > _pointsA;
      std::vector<std::vector<cv::Point2f> > _pointsB;
      _pointsBoard.resize(pointsA.size());
      _pointsA.resize(pointsA.size());
      _pointsB.resize(pointsB.size());

      #pragma omp parallel for
      for (size_t fileIdx =0; fileIdx < _pointsA.size(); fileIdx++){
        // OUT_INFO(std::endl << std::endl << "fileIdx" << std::endl << fileIdx << std::endl);


        std::vector< std::vector< cv::Point2f > > allCornersA;
        for(unsigned int marker_idx = 0; marker_idx < idsA[fileIdx].size(); marker_idx++) {

            std::vector<cv::Point2f> tmp_corners;
            tmp_corners.push_back(pointsA[fileIdx][marker_idx*4+0]);
            tmp_corners.push_back(pointsA[fileIdx][marker_idx*4+1]);
            tmp_corners.push_back(pointsA[fileIdx][marker_idx*4+2]);
            tmp_corners.push_back(pointsA[fileIdx][marker_idx*4+3]);
            allCornersA.push_back(tmp_corners);
            // OUT_INFO(marker_idx << ": corners["<< idsA[fileIdx][marker_idx] <<"] " << tmp_corners[0] <<", "<< tmp_corners[1] <<", "<< tmp_corners[2] <<", "<< tmp_corners[3] );
        }

        // OUT_INFO("imagesA[fileIdx]:" << std::endl << imagesA[fileIdx] << std::endl);

        cv::Mat cv_imageA = cv::imread(imagesA[fileIdx]);
        cv::Mat chess_corners_A, chess_ids_A;
        cv::aruco::interpolateCornersCharuco(allCornersA, idsA[fileIdx], cv_imageA, this->charuco_board,
                                         chess_corners_A, chess_ids_A, cameraMatrixA,
                                         distortionA);

        // OUT_INFO("chess_corners_A:" << std::endl << chess_corners_A << std::endl);
        // OUT_INFO("chess_ids_A:" << std::endl << chess_ids_A << std::endl);


        std::vector< std::vector< cv::Point2f > > allCornersB;
        for(unsigned int marker_idx = 0; marker_idx < idsB[fileIdx].size(); marker_idx++) {

            std::vector<cv::Point2f> tmp_corners;
            tmp_corners.push_back(pointsB[fileIdx][marker_idx*4+0]);
            tmp_corners.push_back(pointsB[fileIdx][marker_idx*4+1]);
            tmp_corners.push_back(pointsB[fileIdx][marker_idx*4+2]);
            tmp_corners.push_back(pointsB[fileIdx][marker_idx*4+3]);
            allCornersB.push_back(tmp_corners);
            // OUT_INFO(marker_idx << ": corners["<< idsB[fileIdx][marker_idx] <<"] " << tmp_corners[0] <<", "<< tmp_corners[1] <<", "<< tmp_corners[2] <<", "<< tmp_corners[3] );
        }

        // OUT_INFO("imagesB[fileIdx]:" << std::endl << imagesB[fileIdx] << std::endl);
        cv::Mat cv_imageB = cv::imread(imagesB[fileIdx]);
        cv::Mat chess_corners_B, chess_ids_B;
        cv::aruco::interpolateCornersCharuco(allCornersB, idsB[fileIdx], cv_imageB, this->charuco_board,
                                         chess_corners_B, chess_ids_B, cameraMatrixB,
                                         distortionB);

        // OUT_INFO("chess_corners_B:" << std::endl << chess_corners_B << std::endl);
        // OUT_INFO("chess_ids_B:" << std::endl << chess_ids_B << std::endl);

        int aIdx = 0;
        int bIdx = 0;

        while (aIdx < chess_ids_A.rows && bIdx < chess_ids_B.rows ){
          if(chess_ids_A.at<int>(aIdx,0) < chess_ids_B.at<int>(bIdx,0)){
            aIdx++;
            continue;
          }

          if(chess_ids_A.at<int>(aIdx,0) > chess_ids_B.at<int>(bIdx,0)){
            bIdx++;
            continue;
          }
          // OUT_INFO("\tCOMMON ID: " << chess_ids_A.at<int>(aIdx,0) << "|" << this->charuco_board->chessboardCorners[chess_ids_A.at<int>(aIdx,0)] << " | " << cv::Point2f(chess_corners_A.at<float>(aIdx,0),chess_corners_A.at<float>(aIdx,1)) <<  " | " << cv::Point2f(chess_corners_B.at<float>(bIdx,0),chess_corners_B.at<float>(bIdx,1)) );

          _pointsBoard[fileIdx].push_back(this->charuco_board->chessboardCorners[chess_ids_A.at<int>(aIdx,0)]);
          _pointsA[fileIdx].push_back(cv::Point2f(chess_corners_A.at<float>(aIdx,0),chess_corners_A.at<float>(aIdx,1)));
          _pointsB[fileIdx].push_back(cv::Point2f(chess_corners_B.at<float>(bIdx,0),chess_corners_B.at<float>(bIdx,1)));

          aIdx++;
          bIdx++;
        }



      }

      int num_points = 0;
      for (int fileIdx = pointsA.size()-1; fileIdx >= 0; fileIdx--){

        // OUT_INFO("_pointsBoard[fileIdx].size:" << _pointsBoard[fileIdx].size() << std::endl);
        if (_pointsBoard[fileIdx].size() >= 3){
          num_points += _pointsBoard[fileIdx].size();
          continue;
        }
        OUT_INFO("Not enough points! Deleting Frame!");
        _pointsBoard.erase(_pointsBoard.begin()+fileIdx);
        _pointsA.erase(_pointsA.begin()+fileIdx);
        _pointsB.erase(_pointsB.begin()+fileIdx);
      }


      cv::Mat rotation_mat;

      OUT_INFO("calibrating extrinsics... "<< _pointsBoard.size() << " frames " << num_points << " and points");
  #if CV_VERSION_MAJOR == 2
      error = cv::stereoCalibrate(_pointsBoard, _pointsB, _pointsA, cameraMatrixB, distortionB, cameraMatrixA, distortionA, imageSize,
                                  rotation_mat, translation, essential, fundamental, termCriteria, cv::CALIB_FIX_INTRINSIC);
  #elif CV_VERSION_MAJOR > 2
      error = cv::stereoCalibrate(_pointsBoard, _pointsB, _pointsA, cameraMatrixB, distortionB, cameraMatrixA, distortionA, imageSize,
                                  rotation_mat, translation, essential, fundamental, cv::CALIB_FIX_INTRINSIC, termCriteria);
  #endif
      OUT_INFO("re-projection error: " << error << std::endl);

      OUT_INFO("Rotation:" << std::endl << rotation_mat);
      OUT_INFO("Translation:" << std::endl << translation);
      OUT_INFO("Essential:" << std::endl << essential);
      OUT_INFO("Fundamental:" << std::endl << fundamental << std::endl);

      // cv::Mat camA_in_refA_rot, camB_in_refB_rot, camB_in_refA_rot;
      // cv::Rodrigues(refA_rot[0], camA_in_refA_rot);
      // cv::Rodrigues(refB_rot[0], camB_in_refB_rot);
      //
      // camB_in_refA_rot =  camA_in_refA_rot * rotation_mat * camB_in_refB_rot.t();
      // cv::Rodrigues(camB_in_refA_rot, rotation);
      // cv::Mat1d translation_mat = refA_trans[0]+camA_in_refA_rot*cv::Mat1d(translation-rotation_mat*camB_in_refB_rot.t()*cv::Mat1d(refB_trans[0]));
      // translation[0] = translation_mat.at<double>(0);
      // translation[1] = translation_mat.at<double>(1);
      // translation[2] = translation_mat.at<double>(2);


      cv::Rodrigues(rotation_mat, rotation);

  }

  void storeCalibration(cv::Vec3d &translation, cv::Vec3d &rotation)
  {
    cv::FileStorage fs;

    fs.open(path + "/" + K2_CALIB_POSE, cv::FileStorage::WRITE);

    if(!fs.isOpened())
    {
      OUT_ERROR("couldn't store calibration data!");
      return;
    }

    fs << K2_CALIB_ROTATION << rotation;
    fs << K2_CALIB_TRANSLATION << translation;
    fs << K2_CALIB_ESSENTIAL << essential;
    fs << K2_CALIB_FUNDAMENTAL << fundamental;

    fs.release();
  }


  bool loadCalibration()
  {

    sensor_msgs::CameraInfo::ConstPtr info_msgA = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("camA_info");

    cameraMatrixA = cv::Mat(3,3,CV_64FC1);
    memcpy(cameraMatrixA.data, info_msgA->K.data(), info_msgA->K.size()*sizeof(double));
    distortionA = cv::Mat(1,info_msgA->D.size(),CV_64FC1);
    memcpy(distortionA.data, info_msgA->D.data(), info_msgA->D.size()*sizeof(double));

    OUT_INFO("Info for A"<<std::endl);
    OUT_INFO("camMatA: " <<cameraMatrixA <<std::endl);
    OUT_INFO("distortA: " <<distortionA <<std::endl);


    sensor_msgs::CameraInfo::ConstPtr info_msgB = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("camB_info");

    cameraMatrixB = cv::Mat(3,3,CV_64FC1);
    memcpy(cameraMatrixB.data, info_msgB->K.data(), info_msgB->K.size()*sizeof(double));
    distortionB = cv::Mat(1,info_msgB->D.size(),CV_64FC1);
    memcpy(distortionB.data, info_msgB->D.data(), info_msgB->D.size()*sizeof(double));
    OUT_INFO("Info for B"<<std::endl);
    OUT_INFO("camMatB: " <<cameraMatrixB <<std::endl);
    OUT_INFO("distortB: " <<distortionB <<std::endl);

    return true;
  }


};


void help(const std::string &path)
{
  std::cout << path << FG_BLUE " [options]" << std::endl
            << FG_GREEN "  board" NO_COLOR ":" << std::endl
            << FG_YELLOW "    'circle<WIDTH>x<HEIGHT>x<SIZE>' (CURRENTLY NOT SUPPORTED) " NO_COLOR "for symmetric circle grid" << std::endl
            << FG_YELLOW "    'acircle<WIDTH>x<HEIGHT>x<SIZE>' (CURRENTLY NOT SUPPORTED) " NO_COLOR "for asymmetric circle grid" << std::endl
            << FG_YELLOW "    'chess<WIDTH>x<HEIGHT>x<SIZE>' (CURRENTLY NOT SUPPORTED)  " NO_COLOR "for chessboard pattern" << std::endl
            << FG_YELLOW "    'charuco<WIDTH>x<HEIGHT>x<SIZE>'   " NO_COLOR "for charuco pattern" << std::endl
            << FG_GREEN "  output path" NO_COLOR ": " FG_YELLOW "'-path <PATH>'" NO_COLOR << std::endl;
}


void publish_tf(std::mutex &mtx, geometry_msgs::TransformStamped &refB_in_refA_msg){
  tf2_ros::TransformBroadcaster br;
  tf2_ros::Buffer tfBuffer;

  ros::Rate rate(10.0);
  while (ros::ok()){
    // ROS_INFO_STREAM("thread loop "<< refB_in_refA_msg.header.stamp);
    // refB_in_refA_msg.header.frame_id = camA_tf_frame;
    // refB_in_refA_msg.child_frame_id = boardA_tf_frame;
    // refB_in_refA_msg.transform.translation.x = tvecA[0];
    // refB_in_refA_msg.transform.translation.y = tvecA[1];
    // refB_in_refA_msg.transform.translation.z = tvecA[2];
    // double angle = sqrt(pow(rvecA[0],2)+pow(rvecA[1],2)+pow(rvecA[2],2));
    // refB_in_refA_msg.transform.rotation.w = cos(0.5*angle);
    // refB_in_refA_msg.transform.rotation.x = rvecA[0]/angle * sin(0.5*angle);
    // refB_in_refA_msg.transform.rotation.y = rvecA[1]/angle * sin(0.5*angle);
    // refB_in_refA_msg.transform.rotation.z = rvecA[2]/angle * sin(0.5*angle);
    mtx.lock();
    refB_in_refA_msg.header.stamp = ros::Time::now();
    br.sendTransform(refB_in_refA_msg);
    mtx.unlock();
    rate.sleep();
  }
}

int main(int argc, char **argv)
{
#if EXTENDED_OUTPUT
  ROSCONSOLE_AUTOINIT;
  if(!getenv("ROSCONSOLE_FORMAT"))
  {
    ros::console::g_formatter.tokens_.clear();
    ros::console::g_formatter.init("[${severity}] ${message}");
  }
#endif

  Board board_type = CHARUCO;
  int aruco_dict_id = 8;
  cv::Size boardDims = cv::Size(7, 6);
  float boardSize = 0.108;
  std::string path = ".";

  ros::init(argc, argv, "camera_ext_calib", ros::init_options::AnonymousName);
  ros::NodeHandle nh("~");

  if(!ros::ok())
  {
    return 0;
  }

  for(int argI = 1; argI < argc; ++ argI)
  {
    std::string arg(argv[argI]);

    if(arg == "--help" || arg == "--h" || arg == "-h" || arg == "-?" || arg == "--?")
    {
      help(argv[0]);
      ros::shutdown();
      return 0;
    }
    else if(arg.find("circle") == 0 && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
    {
      // board_type  = CIRCLE;
      // const size_t start = 6;
      // const size_t leftX = arg.find('x');
      // const size_t rightX = arg.rfind('x');
      // const size_t end = arg.size();
      //
      // int width = atoi(arg.substr(start, leftX - start).c_str());
      // int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
      // boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
      // boardDims = cv::Size(width, height);
      ROS_FATAL("Circle boards are currently not supported");
    }
    else if(arg.find("acircle") == 0 && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
    {
      // board_type  = ACIRCLE;
      // const size_t start = 7;
      // const size_t leftX = arg.find('x');
      // const size_t rightX = arg.rfind('x');
      // const size_t end = arg.size();
      //
      // int width = atoi(arg.substr(start, leftX - start).c_str());
      // int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
      // boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
      // boardDims = cv::Size(width, height);
      ROS_FATAL("Asymmetric circle boards are currently not supported");
    }
    else if(arg.find("chess") == 0 && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
    {
      // board_type  = CHESS;
      // const size_t start = 5;
      // const size_t leftX = arg.find('x');
      // const size_t rightX = arg.rfind('x');
      // const size_t end = arg.size();
      //
      // int width = atoi(arg.substr(start, leftX - start).c_str());
      // int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
      // boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
      // boardDims = cv::Size(width, height);
      ROS_FATAL("Chess are currently not supported");
    }
    else if(arg.find("charuco") == 0 && arg.find('x') != arg.rfind('x') && arg.rfind('x') != std::string::npos)
    {
      board_type  = CHARUCO;
      const size_t start = 7;
      const size_t leftX = arg.find('x');
      const size_t rightX = arg.rfind('x');
      const size_t end = arg.size();

      int width = atoi(arg.substr(start, leftX - start).c_str());
      int height = atoi(arg.substr(leftX + 1, rightX - leftX + 1).c_str());
      boardSize = atof(arg.substr(rightX + 1, end - rightX + 1).c_str());
      boardDims = cv::Size(width, height);
    }
    else if(arg == "-path" && ++argI < argc)
    {
      arg = argv[argI];
      struct stat fileStat;
      if(stat(arg.c_str(), &fileStat) == 0 && S_ISDIR(fileStat.st_mode))
      {
        path = arg;
      }
      else
      {
        OUT_ERROR("Unknown path: " << arg);
        help(argv[0]);
        ros::shutdown();
        return 0;
      }
    }
    else
    {
      ROS_FATAL("UNKNOWN ARGUMENT");
    }
  }

  OUT_INFO("Start settings:" << std::endl
           << "       Board: " FG_CYAN << (board_type == CHESS ? "chess" : (board_type == CIRCLE ? "circle" : "charuco") ) << NO_COLOR << std::endl
           << "  Dimensions: " FG_CYAN << boardDims.width << " x " << boardDims.height << NO_COLOR << std::endl
           << "  Field size: " FG_CYAN << boardSize << NO_COLOR << std::endl
           << "        Path: " FG_CYAN << path << NO_COLOR << std::endl);

  if(!ros::master::check())
  {
    OUT_ERROR("checking ros master failed.");
    return -1;
  }

  std::string camA_tf_frame, camB_tf_frame, boardA_tf_frame, boardB_tf_frame, referenceA_tf_frame, referenceB_tf_frame;
  nh.param<std::string>("camA_tf_frame", camA_tf_frame, "camA");
  nh.param<std::string>("camB_tf_frame", camB_tf_frame, "camB");
  nh.param<std::string>("boardA_tf_frame", boardA_tf_frame, "boardA");
  nh.param<std::string>("boardB_tf_frame", boardB_tf_frame, "boardB");
  nh.param<std::string>("referenceA_tf_frame", referenceA_tf_frame, camA_tf_frame);
  nh.param<std::string>("referenceB_tf_frame", referenceB_tf_frame, camB_tf_frame);
  referenceA_tf_frame = camA_tf_frame;
  // referenceB_tf_frame = camB_tf_frame;



  // tf2_ros::TransformBroadcaster br;
  // tf2_ros::Buffer tfBuffer;
  tf2::Stamped<tf2::Transform> refB_in_refA;
  geometry_msgs::TransformStamped refB_in_refA_msg;
  refB_in_refA_msg.header.frame_id = referenceA_tf_frame;
  refB_in_refA_msg.child_frame_id = referenceB_tf_frame;
  refB_in_refA_msg.transform.rotation.w = 1.0;

  ROS_INFO_STREAM("SETUP FRAME FROM " << refB_in_refA_msg.header.frame_id  << " TO " << refB_in_refA_msg.child_frame_id );
  std::mutex tf_mutex;
  std::thread tf_thread(publish_tf, std::ref(tf_mutex), std::ref(refB_in_refA_msg));

  cv::Vec3d rotation, translation;

  char input = 'r';
  while(input != 'q'){

    if(input =='r'){


      CameraCalibration calib(nh, path, cv::Size(1920,1080), board_type, boardDims, boardSize, aruco_dict_id);
      OUT_INFO("restoring files...");
      bool ret = calib.restore();

      if (!ret){
        OUT_ERROR("COULDN'T RESTORE THE DATA");
      }else{
        OUT_INFO("starting calibration...");
        calib.calibrate(translation, rotation);

        tf_mutex.lock();
        opencv_to_tf2(translation, rotation, refB_in_refA);
        tf2::convert(refB_in_refA,refB_in_refA_msg);
        refB_in_refA_msg.header.frame_id = referenceA_tf_frame;
        refB_in_refA_msg.child_frame_id = referenceB_tf_frame;
        tf_mutex.unlock();

        // transformStamped.transform.translation.x = tvecA[0];
        // transformStamped.transform.translation.y = tvecA[1];
        // transformStamped.transform.translation.z = tvecA[2];
        // double angle = sqrt(pow(rvecA[0],2)+pow(rvecA[1],2)+pow(rvecA[2],2));
        // transformStamped.transform.rotation.w = cos(0.5*angle);
        // transformStamped.transform.rotation.x = rvecA[0]/angle * sin(0.5*angle);
        // transformStamped.transform.rotation.y = rvecA[1]/angle * sin(0.5*angle);
        // transformStamped.transform.rotation.z = rvecA[2]/angle * sin(0.5*angle);
      }

    }else{
      std::cout << "Unknown command: " << input << std::endl;
    }

    std::cout << "Would you like to quit (q) or re-calibrate (r)" << std::endl;
    std::cin >> input;
  }
  ros::shutdown();
  tf_thread.join();

  return 0;
}
