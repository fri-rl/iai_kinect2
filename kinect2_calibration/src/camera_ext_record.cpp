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
#include <map>
#include <mutex>
#include <thread>

#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
// #include <opencv2/aruco.hpp>

// If OpenCV4
#if CV_VERSION_MAJOR > 3
#include <opencv2/imgcodecs/legacy/constants_c.h>
#endif

#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <math.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Transform.h>
// #include <tf2/tf2_geometry_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// #include <kinect2_calibration/kinect2_calibration_definitions.h>
// #include <kinect2_bridge/kinect2_definitions.h>

enum Board
{
  CHESS,
  CIRCLE,
  ACIRCLE,
  CHARUCO
};

class Recorder
{
private:
  const Board board_type;
  int circleFlags;

  const std::string path;

  // const std::string topicA, topicB;
  std::mutex lock;

  bool update;
  bool foundA, foundB;
  cv::Mat cv_imageA, cv_imageB;


  size_t frame;
  std::vector<int> params;

  // std::vector<cv::Point3f> board;
  std::vector<cv::Point2f> pointsA, pointsB;
  std::vector<int> idsA, idsB;



  cv::Mat cameraMatrixA, distortionA;
  cv::Mat cameraMatrixB, distortionB;

  cv::Ptr<cv::aruco::CharucoBoard> charuco_board;
  cv::Ptr<cv::aruco::Board> aruco_board;


 std::vector< std::vector< cv::Point2f > > aruco_corners_A, aruco_corners_B;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ABSyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence ABSyncPolicy(10) is a policy with queuesize 10

  ros::NodeHandle nh;
  ros::AsyncSpinner spinner;
  image_transport::ImageTransport it;
  image_transport::SubscriberFilter *subImageA, *subImageB;
  message_filters::Synchronizer<ABSyncPolicy> *sync;


  std::string camA_tf_frame, camB_tf_frame, boardA_tf_frame, boardB_tf_frame, referenceA_tf_frame, referenceB_tf_frame;
  tf2_ros::TransformBroadcaster br;
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  tf2::Stamped<tf2::Transform> cameraA_in_ref;
  tf2::Stamped<tf2::Transform> cameraB_in_ref;

public:
  Recorder(const ros::NodeHandle &nh, const std::string &path, const Board board_type, cv::Ptr<cv::aruco::CharucoBoard> board)
    : board_type(board_type), path(path), update(false), foundA(false), foundB(false), frame(0), nh(nh), spinner(0), it(nh), tfListener(tfBuffer)
    {

    //##################################################################
    //############# Commented out since currently not supported ########
    //##################################################################
    //
    // if(board_type == CIRCLE)
    // {
    //   circleFlags = cv::CALIB_CB_SYMMETRIC_GRID + cv::CALIB_CB_CLUSTERING;
    // }
    // else if (board_type == ACIRCLE)
    // {
    //   circleFlags = cv::CALIB_CB_ASYMMETRIC_GRID + cv::CALIB_CB_CLUSTERING;
    // }
    //
    // board.resize(boardDims.width * boardDims.height);
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

    params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    params.push_back(9);

    if (board_type == CHARUCO){
      // create charuco board object
      charuco_board = board;
      aruco_board = charuco_board.staticCast<cv::aruco::Board>();
    }

    frame = find_next_frame_id();
    ROS_INFO_STREAM("NEXT_FRAME_ID: " << frame);


        nh.param<std::string>("camA_tf_frame", camA_tf_frame, "camA");
        nh.param<std::string>("camB_tf_frame", camB_tf_frame, "camB");
        nh.param<std::string>("boardA_tf_frame", boardA_tf_frame, "boardA");
        nh.param<std::string>("boardB_tf_frame", boardB_tf_frame, "boardB");
        nh.param<std::string>("referenceA_tf_frame", referenceA_tf_frame, camA_tf_frame);
        nh.param<std::string>("referenceB_tf_frame", referenceB_tf_frame, camB_tf_frame);

  }
  ~Recorder()
  {
  }

  int find_next_frame_id(){

    DIR *dp;
    struct dirent *dirp;

    if((dp  = opendir(path.c_str())) ==  NULL)
    {
      ROS_ERROR_STREAM("Error opening: " << path);
      return false;
    }

    int frame_id = -1;
    while((dirp = readdir(dp)) != NULL)
    {
      if(dirp->d_type != DT_REG)
      {
        continue;
      }

      std::string filename = dirp->d_name;
      int file_frame_id;
      try {
        std::regex re("^(\\d+)_.*\\.yaml");
        std::smatch match;
        if (std::regex_search(filename, match, re) && match.size() > 1) {
          file_frame_id = std::stoi(match.str(1));
          if (file_frame_id > frame_id){
            frame_id = file_frame_id;
          }
        }
      } catch (std::regex_error& e) {
        ROS_ERROR_STREAM("RE Error: " << e.what());
      } catch (std::invalid_argument& e) {
        ROS_ERROR_STREAM("STOI Error: " << e.what());
      }
    }
    closedir(dp);

    return frame_id + 1;
  }

  void run()
  {

    loadCalibration();

    startRecord();

    display();

    stopRecord();
  }

private:
  void startRecord()
  {
    ROS_INFO_STREAM("Controls:" << std::endl
             << "   [ESC, q] - Exit" << std::endl
             << " [SPACE, s] - Save current frame");

    // image_transport::TransportHints hints("compressed");

    // subImageA= new image_transport::SubscriberFilter(it, ros::names::remap("camA_image"), 4, hints);
    // subImageB = new image_transport::SubscriberFilter(it, ros::names::remap("camB_image"), 4, hints);
    subImageA= new image_transport::SubscriberFilter(it, ros::names::remap("camA_image"), 4);
    subImageB = new image_transport::SubscriberFilter(it, ros::names::remap("camB_image"), 4);

    sync = new message_filters::Synchronizer<ABSyncPolicy>(ABSyncPolicy(10), *subImageA, *subImageB);
    sync->registerCallback(boost::bind(&Recorder::callback, this, _1, _2));

    spinner.start();
  }

  void stopRecord()
  {
    spinner.stop();

    delete sync;
    delete subImageA;
    delete subImageB;
  }


  void callback(const sensor_msgs::Image::ConstPtr ros_imageA, const sensor_msgs::Image::ConstPtr ros_imageB)
  {
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
    // detectorParams->maxErroneousBitsInBorderRate = 0.5;
    // cv::aruco::DetectorParameters baam;
    // ROS_INFO_STREAM("WHYYYY "<< cv::aruco::DetectorParameters::aprilTagQuadSigma)
    // baam.aprilTagQuadSigma = 0.8;

    // detectorParams->aprilTagQuadSigma = 0.8;
    // detectorParams->detectInvertedMarker = true;
    detectorParams->markerBorderBits =1; 
    detectorParams->minOtsuStdDev = 0.5;

    detectorParams->maxErroneousBitsInBorderRate = 0.6;
    detectorParams->errorCorrectionRate = 0.9;
    bool refineStrategy = true;

    std::vector<cv::Point2f> pointsA, pointsB;
    std::vector<int> idsA, idsB;

    std::vector< std::vector< cv::Point2f > > aruco_corners_A, aruco_corners_rejected_A;
    std::vector< std::vector< cv::Point2f > > aruco_corners_B, aruco_corners_rejected_B;

    cv::Mat cv_imageA, cv_imageB;
    bool foundA = false;
    bool foundB = false;

    readImage(ros_imageA, cv_imageA);
    readImage(ros_imageB, cv_imageB);



    geometry_msgs::TransformStamped transformStamped;
    transformStamped = tfBuffer.lookupTransform(referenceA_tf_frame, camA_tf_frame, ros::Time(0));
    tf2::convert(transformStamped, cameraA_in_ref);

    transformStamped = tfBuffer.lookupTransform(referenceB_tf_frame, camB_tf_frame, ros::Time(0));
    tf2::convert(transformStamped, cameraB_in_ref);



    //##################################################################
    //############# Commented out since currently not supported ########
    //##################################################################
    //
    // if(board_type == CIRCLE)
    // {
    //   foundA = cv::findCirclesGrid(cv_imageA, boardDims, pointsA, circleFlags);
    //   foundB = cv::findCirclesGrid(cv_imageB, boardDims, pointsB, circleFlags);
    // }
    // else if(board_type == CHESS)
    // {
    //   foundA = cv::findChessboardCorners(cv_imageA, boardDims, pointsA, cv::CALIB_CB_FAST_CHECK);
    //   foundB = cv::findChessboardCorners(cv_imageB, boardDims, pointsB, cv::CALIB_CB_FAST_CHECK);
    //   if(foundA)
    //   {
    //     cv::cornerSubPix(cv_imageA, pointsA, cv::Size(11, 11), cv::Size(-1, -1), termCriteria);
    //   }
    //   if(foundB)
    //   {
    //     cv::cornerSubPix(cv_imageB, pointsB, cv::Size(11, 11), cv::Size(-1, -1), termCriteria);
    //   }
    // }
    if(board_type == CHARUCO)
    {


      // ##########################################
      // ########## MAKE A FUNCTION FOR THIS ######
      // ##########################################

      // detect markers
      cv::aruco::detectMarkers(cv_imageA, charuco_board->dictionary, aruco_corners_A, idsA, detectorParams, aruco_corners_rejected_A, cameraMatrixA, distortionA );

      // refind strategy to detect more markers
      if(refineStrategy) {
        cv::aruco::refineDetectedMarkers(cv_imageA, aruco_board, aruco_corners_A, idsA, aruco_corners_rejected_A, cameraMatrixA, distortionA);
      }

      foundA = idsA.size() > 0;

      if (foundA){
          for(size_t i = 0; i < aruco_corners_A.size(); ++i){
              for(size_t j = 0; j < aruco_corners_A[i].size(); ++j){
                pointsA.push_back(aruco_corners_A[i][j]);
              }
          }
      }


      // ##########################################
      // ########## SEE ABOVE!!!! ######
      // ##########################################

      // detect markers
      cv::aruco::detectMarkers(cv_imageB, charuco_board->dictionary, aruco_corners_B, idsB, detectorParams, aruco_corners_rejected_B, cameraMatrixB, distortionB);

      // refind strategy to detect more markers
      if(refineStrategy) {
        cv::aruco::refineDetectedMarkers(cv_imageB, aruco_board, aruco_corners_B, idsB, aruco_corners_rejected_B, cameraMatrixB, distortionB);
      }

      foundB = idsB.size() > 0;

      if (foundB){
          for(size_t i = 0; i < aruco_corners_B.size(); ++i){
              for(size_t j = 0; j < aruco_corners_B[i].size(); ++j){
                pointsB.push_back(aruco_corners_B[i][j]);
              }
          }
      }

    }

    lock.lock();
    this->cv_imageA = cv_imageA;
    this->cv_imageB = cv_imageB;
    this->foundA = foundA;
    this->foundB = foundB;
    this->pointsA = pointsA;
    this->pointsB = pointsB;
    this->idsA = idsA;
    this->idsB = idsB;
    this->aruco_corners_A = aruco_corners_A;
    this->aruco_corners_B = aruco_corners_B;

    update = true;
    lock.unlock();
  }

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

  void display()
  {
    std::vector<cv::Point2f> pointsA, pointsB;
    std::vector<int> idsA, idsB;
    cv::Mat cv_imageA, cv_imageB;
    cv::Mat dispA, dispB;
    bool foundA = false;
    bool foundB = false;
    bool save = false;
    bool running = true;

    std::vector< std::vector< cv::Point2f > > aruco_corners_A, aruco_corners_rejected_A;
    cv::Mat charuco_corners_A, charuco_ids_A;

    std::vector< std::vector< cv::Point2f > > aruco_corners_B, aruco_corners_rejected_B;
    cv::Mat charuco_corners_B, charuco_ids_B;



    std::chrono::milliseconds duration(1);
    while(!update && ros::ok())
    {
      std::this_thread::sleep_for(duration);
    }

    for(; ros::ok() && running;)
    {
      if(update)
      {
        lock.lock();
        cv_imageA = this->cv_imageA;
        cv_imageB = this->cv_imageB;
        foundA = this->foundA;
        foundB = this->foundB;
        pointsA = this->pointsA;
        pointsB = this->pointsB;
        idsA = this->idsA;
        idsB = this->idsB;
        aruco_corners_A = this->aruco_corners_A;
        aruco_corners_B = this->aruco_corners_B;

        cv::Vec3d rvecA, tvecA, rvecB, tvecB;
        cv::Vec3d ref_rvecA, ref_tvecA, ref_rvecB, ref_tvecB;


        update = false;
        lock.unlock();

        // ##########################################
        // ########## MAKE A FUNCTION FOR THIS ######
        // ##########################################

        cv::cvtColor(cv_imageA, dispA, CV_GRAY2BGR);
        if (board_type == CHARUCO){
          if (idsA.size() > 0){
            cv::aruco::drawDetectedMarkers(dispA, aruco_corners_A);
            cv::aruco::interpolateCornersCharuco(aruco_corners_A, idsA, cv_imageA, charuco_board, charuco_corners_A, charuco_ids_A);


            if (charuco_corners_A.total() > 0){
              cv::aruco::drawDetectedCornersCharuco(dispA, charuco_corners_A, charuco_ids_A);

              bool valid = cv::aruco::estimatePoseCharucoBoard(charuco_corners_A, charuco_ids_A, charuco_board, cameraMatrixA, distortionA, rvecA, tvecA);
              if(valid){
                  cv::aruco::drawAxis(dispA, cameraMatrixA, distortionA, rvecA, tvecA, charuco_board->getSquareLength());
                  std::ostringstream pose;
                  pose << rvecA << " " << tvecA;
                  cv::putText(dispA, pose.str(), cv::Point(5,75), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,200), 4);

                  tf2::Stamped<tf2::Transform> board_in_camera;
                  geometry_msgs::TransformStamped board_in_camera_msg;
                  opencv_to_tf2(tvecA, rvecA, board_in_camera);

                  tf2::convert(board_in_camera, board_in_camera_msg);
                  board_in_camera_msg.header.stamp = ros::Time::now();
                  board_in_camera_msg.header.frame_id = camA_tf_frame;
                  board_in_camera_msg.child_frame_id = boardA_tf_frame;
                  br.sendTransform(board_in_camera_msg);

                  tf2::Transform board_in_ref = cameraA_in_ref*board_in_camera;

                  tf2_to_opencv(board_in_ref, ref_tvecA, ref_rvecA);

                  std::ostringstream board_in_reference_str;
                  board_in_reference_str << ref_rvecA << " " << ref_tvecA;
                  cv::putText(dispA, board_in_reference_str.str(), cv::Point(5,150), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,200,0), 4);

              }
            }
          }
        }
        // ONLY CHARUCO SUPPORTED SO FAR
        // else{
        //   cv::drawChessboardCorners(dispA, boardDims, pointsA, foundA);
        // }

        cv::resize(dispA, dispA, cv::Size(), 0.3, 0.3);



        // ##########################################
        // ########## SEE ABOVE!!!! ######
        // ##########################################

        cv::cvtColor(cv_imageB, dispB, CV_GRAY2BGR);
        if (board_type == CHARUCO){
          if (idsB.size() > 0){
            cv::aruco::drawDetectedMarkers(dispB, aruco_corners_B);
            cv::aruco::interpolateCornersCharuco(aruco_corners_B, idsB, cv_imageB, charuco_board, charuco_corners_B, charuco_ids_B);

            if (charuco_corners_B.total() > 0){
              cv::aruco::drawDetectedCornersCharuco(dispB, charuco_corners_B, charuco_ids_B);

              bool valid = cv::aruco::estimatePoseCharucoBoard(charuco_corners_B, charuco_ids_B, charuco_board, cameraMatrixB, distortionB, rvecB, tvecB);
              if(valid){
                  cv::aruco::drawAxis(dispB, cameraMatrixB, distortionB, rvecB, tvecB, charuco_board->getSquareLength());
                  std::ostringstream pose;
                  pose << rvecB << " " << tvecB;
                  cv::putText(dispB, pose.str(), cv::Point(5,75), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,200), 4);

                  tf2::Stamped<tf2::Transform> board_in_camera;
                  geometry_msgs::TransformStamped board_in_camera_msg;
                  opencv_to_tf2(tvecB, rvecB, board_in_camera);

                  tf2::convert(board_in_camera, board_in_camera_msg);
                  board_in_camera_msg.header.stamp = ros::Time::now();
                  board_in_camera_msg.header.frame_id = camB_tf_frame;
                  board_in_camera_msg.child_frame_id = boardB_tf_frame;
                  br.sendTransform(board_in_camera_msg);

                  tf2::Transform board_in_ref = cameraB_in_ref*board_in_camera;

                  tf2_to_opencv(board_in_ref, ref_tvecB, ref_rvecB);

                  std::ostringstream board_in_reference_str;
                  board_in_reference_str << ref_rvecB << " " << ref_tvecB;
                  cv::putText(dispB, board_in_reference_str.str(), cv::Point(5,150), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,200,0), 4);


              }
            }
          }
        }
        // ONLY CHARUCO SUPPORTED SO FAR
        // else{
        //   cv::drawChessboardCorners(dispB, boardDims, pointsB, foundB);
        // }

        cv::resize(dispB, dispB, cv::Size(), 0.3, 0.3);

        // ROS_INFO_STREAM("Charuco board pose:" << std::endl << "\tA: " << tvecA << "; " << rvecA << std::endl << "\tB: " << tvecB << "; " << rvecB << std::endl);

      }

      cv::imshow("camera A", dispA);
      cv::imshow("camera B", dispB);

      int key = cv::waitKey(10);
      switch(key & 0xFF)
      {
      case ' ':
      case 's':
        save = true;
        break;
      case 27:
      case 'q':
        running = false;
        break;
      }

      if(save){
        if( foundA && foundB){
          std::string base = get_store_filename_base();
          ROS_INFO_STREAM("storing frame: " << base);
          store(base, cv_imageA, cv_imageB, pointsA, pointsB, cameraA_in_ref , cameraB_in_ref, idsA, idsB);
        }
        save = false;
      }
    }
    cv::destroyAllWindows();
    cv::waitKey(100);
  }

  void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const
  {
    cv_bridge::CvImageConstPtr pCvImage;
    // ROS_INFO_STREAM("ENCODING " << msgImage->encoding);
    // pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
    pCvImage = cv_bridge::toCvShare(msgImage, sensor_msgs::image_encodings::MONO8);
    pCvImage->image.copyTo(image);
    // image = cv_bridge::toCvCopy(msgImage, sensor_msgs::image_encodings::MONO8);
  }

  std::string get_store_filename_base(){
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(4) << frame++;
    const std::string frameNumber(oss.str());
    std::string base = path + "/" + frameNumber + "_ext";
    return base;
  }

  void store(std::string base, const cv::Mat &cv_imageA, const cv::Mat &cv_imageB, const std::vector<cv::Point2f> &pointsA, const std::vector<cv::Point2f> &pointsB, const tf2::Stamped<tf2::Transform> &transformA, const tf2::Stamped<tf2::Transform> &transformB , const std::vector<int> &idsA, const std::vector<int> &idsB)
  {
    cv::imwrite(base + "_A.png", cv_imageA, params);
    cv::imwrite(base + "_B.png", cv_imageB, params);

    cv::Vec3d ref_rvecA, ref_tvecA, ref_rvecB, ref_tvecB;
    tf2_to_opencv(transformA, ref_tvecA, ref_rvecA);
    tf2_to_opencv(transformB, ref_tvecB, ref_rvecB);

    cv::FileStorage file(base + ".yaml", cv::FileStorage::WRITE);
    file << "pointsA" << pointsA;
    file << "pointsB" << pointsB;
    file << "idsA" << idsA;
    file << "idsB" << idsB;
    file << "refA_trans" << ref_tvecA;
    file << "refA_rot" << ref_rvecA;
    file << "refB_trans" << ref_tvecB;
    file << "refB_rot" << ref_rvecB;
  }


  bool loadCalibration()
  {

    sensor_msgs::CameraInfo::ConstPtr info_msgA = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("camA_info");
    cameraMatrixA = cv::Mat(3,3,CV_64FC1);
    memcpy(cameraMatrixA.data, info_msgA->K.data(), info_msgA->K.size()*sizeof(double));
    distortionA = cv::Mat(1,info_msgA->D.size(),CV_64FC1);
    memcpy(distortionA.data, info_msgA->D.data(), info_msgA->D.size()*sizeof(double));

    ROS_INFO_STREAM("Info for A"<<std::endl);
    ROS_INFO_STREAM("camMatA: " <<cameraMatrixA <<std::endl);
    ROS_INFO_STREAM("distortA: " <<distortionA <<std::endl);


    sensor_msgs::CameraInfo::ConstPtr info_msgB = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("camB_info");
    cameraMatrixB = cv::Mat(3,3,CV_64FC1);
    memcpy(cameraMatrixB.data, info_msgB->K.data(), info_msgB->K.size()*sizeof(double));
    distortionB = cv::Mat(1,info_msgB->D.size(),CV_64FC1);
    memcpy(distortionB.data, info_msgB->D.data(), info_msgB->D.size()*sizeof(double));
    ROS_INFO_STREAM("Info for B"<<std::endl);
    ROS_INFO_STREAM("camMatB: " <<cameraMatrixB <<std::endl);
    ROS_INFO_STREAM("distortB: " <<distortionB <<std::endl);

    return true;
  }

};


void help(const std::string &path)
{
  std::cout << path << " [options]" << std::endl
            << "  board:" << std::endl
            << "    'circle<WIDTH>x<HEIGHT>x<SIZE>' (CURRENTLY NOT SUPPORTED) for symmetric circle grid" << std::endl
            << "    'acircle<WIDTH>x<HEIGHT>x<SIZE>'(CURRENTLY NOT SUPPORTED) for asymmetric circle grid" << std::endl
            << "    'chess<WIDTH>x<HEIGHT>x<SIZE>' (CURRENTLY NOT SUPPORTED)  for chessboard pattern" << std::endl
            << "    'charuco<WIDTH>x<HEIGHT>x<SIZE>'   for charuco pattern" << std::endl
            << "  output path: '-path <PATH>'" << std::endl;
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
  std::string path = ".";

  std::map<std::string, cv::aruco::PREDEFINED_DICTIONARY_NAME> aruco_ids;
  aruco_ids["4x4_50"] = cv::aruco::DICT_4X4_50;
  aruco_ids["6x6_50"] = cv::aruco::DICT_6X6_50;
  aruco_ids["6x6_250"] = cv::aruco::DICT_6X6_250;

  ros::init(argc, argv, "camera_ext_record", ros::init_options::AnonymousName);
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
      board_type  = CIRCLE;
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
      board_type  = ACIRCLE;
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
      board_type  = CHESS;
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
    else if(arg.find("charuco") == 0)
    {
      board_type  = CHARUCO;
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
        ROS_ERROR_STREAM("Unknown path: " << arg);
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

  ROS_INFO_STREAM("Start settings:" << std::endl
           << "      Board: " << (board_type == CHESS ? "chess" : (board_type == CIRCLE ? "circle" : "charuco") ) << std::endl
           << "       Path: " << path << std::endl
            );

  if(!ros::master::check())
  {
    ROS_ERROR_STREAM("checking ros master failed.");
    return -1;
  }



  cv::Ptr<cv::aruco::CharucoBoard> board;

  if (board_type == CHARUCO){

    std::string aruco_desc;
    int charuco_rows, charuco_cols;
    double charuco_aruco_size, charuco_chess_size;


    nh.param<std::string>("charuco/dict", aruco_desc, "6x6_50");
    nh.param<int>("charuco/rows", charuco_rows, 3);
    nh.param<int>("charuco/cols", charuco_cols, 4);
    nh.param<double>("charuco/aruco_size", charuco_aruco_size, 0.05);
    nh.param<double>("charuco/chess_size", charuco_chess_size, 0.065);


    ROS_INFO_STREAM("charuco/dict " << aruco_desc << " " << aruco_ids.at(aruco_desc) << " ");
    ROS_INFO_STREAM("charuco/rows " << charuco_rows);
    ROS_INFO_STREAM("charuco/cols " << charuco_cols);
    ROS_INFO_STREAM("charuco/aruco_size " << charuco_aruco_size);
    ROS_INFO_STREAM("charuco/chess_size " << charuco_chess_size);

    board = cv::aruco::CharucoBoard::create(charuco_rows, charuco_cols, charuco_chess_size, charuco_aruco_size, cv::aruco::getPredefinedDictionary(aruco_ids.at(aruco_desc)));
  }


  Recorder recorder(nh, path, board_type, board);

  ROS_INFO_STREAM("starting recorder...");
  recorder.run();

  ROS_INFO_STREAM("stopped recording...");

  return 0;
}
