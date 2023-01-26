#include "lk_feature_tracker.h"

#include <numeric>
#include <vector>

#include <glog/logging.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <ros/ros.h>

using namespace cv;
using namespace cv::xfeatures2d;

/**
   LK feature tracker Constructor.
*/
LKFeatureTracker::LKFeatureTracker() {
  // Feel free to modify if you want!
  cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
}

LKFeatureTracker::~LKFeatureTracker() {
  // Feel free to modify if you want!
  cv::destroyWindow(window_name_);
}

/** TODO This is the main tracking function, given two images, it detects,
 * describes and matches features.
 * We will be modifying this function incrementally to plot different figures
 * and compute different statistics.
 @param[in] frame Current image frame
*/
void LKFeatureTracker::trackFeatures(const cv::Mat& frame) {
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  DELIVERABLE 7 | Feature Tracking: Lucas-Kanade Tracker
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //
  // For this part, you will need to:
  //
  //   1. Using OpenCV’s documentation and the C++ API for the LK tracker, track
  //   features for the video sequences we provided you by using the Harris
  //   corner detector (like here). Show the feature tracks at a given frame
  //   extracted when using the Harris corners (consider using the 'show'
  //   function below)
  //
  //   Hint 1: take a look at cv::goodFeaturesToTrack and
  //   cv::calcOpticalFlowPyrLK
  //
  //   2. Add an extra entry in the table you made previously for the Harris +
  //   LK tracker
  //
  //   Note: LKFeatureTracker does not inherit from the base tracker like other
  //   feature trackers, so you need to also implement the statistics gathering
  //   code right here.
  //
  // ~~~~ begin solution
  //

  // Convert to grayscale
  cv::Mat curr_frame_gray;
  cv::cvtColor(frame, curr_frame_gray, cv::COLOR_BGR2GRAY);

  if (p0_.empty()) {
    // First frame, detect features
    cv::goodFeaturesToTrack(curr_frame_gray, p0_, 100, 0.01, 10);

    // Set the previous frame to the current frame
    prev_frame_ = curr_frame_gray.clone();

    // Initialize mask
    mask_ = cv::Mat::zeros(frame.size(), frame.type());

    return;
  }

  // Extract features using goodFeaturesToTrack
  std::vector<cv::Point2f> curr_features;
  cv::goodFeaturesToTrack(curr_frame_gray, curr_features, 100, 0.01, 10);

  // Track features using calcOpticalFlowPyrLK
  std::vector<cv::Point2f> p1;
  std::vector<uchar> status;
  std::vector<float> error;
  TermCriteria criteria =
      TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
  cv::calcOpticalFlowPyrLK(prev_frame_, curr_frame_gray, p0_, p1, status, error,
                           Size(15, 15), 2, criteria);

  // Use status to copy over only matched features
  std::vector<cv::Point2f> good_old;
  std::vector<cv::Point2f> good_new;
  for (size_t i = 0; i < status.size(); i++) {
    if (status[i]) {
      good_old.push_back(p0_[i]);
      good_new.push_back(p1[i]);
    }
  }

  // Display the image
  show(frame, good_old, good_new);

  // Set the previous frame to the current frame
  prev_frame_ = curr_frame_gray.clone();

  // Set the previous features to the current features
  p0_ = good_new;

  //
  // ~~~~ end solution
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //                             end deliverable 7
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}

/** TODO Display image with tracked features from prev to curr on the image
 * corresponding to 'frame'
 * @param[in] frame The current image frame, to draw the feature track on
 * @param[in] prev The previous set of keypoints
 * @param[in] curr The set of keypoints for the current frame
 */
void LKFeatureTracker::show(const cv::Mat& frame,
                            std::vector<cv::Point2f>& prev,
                            std::vector<cv::Point2f>& curr) {
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // ~~~~ begin solution
  //
  //
  //     Hint: look at cv::line and cv::cirle functions.
  //     Hint 2: use imshow to display the image

  // Draw the features
  for (size_t i = 0; i < prev.size(); i++) {
    // cv::circle(frame, prev[i], 3, cv::Scalar(0, 255, 0), -1);
    cv::circle(mask_, curr[i], 3, cv::Scalar(0, 0, 255), -1);
    cv::line(mask_, curr[i], prev[i], cv::Scalar(255, 0, 0), 2);
  }

  // Add the mask to the image
  cv::add(frame, mask_, frame);

  // Display the image
  cv::imshow(window_name_, frame);
  cv::waitKey(10);

  //
  // ~~~~ end solution
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}

/** Compute Inlier Mask out of the given matched keypoints.
 @param[in] pts1 List of keypoints detected on the first image.
 @param[in] pts2 List of keypoints detected on the second image.
 @param[out] inlier_mask Mask indicating inliers (1) from outliers (0).
*/
void LKFeatureTracker::inlierMaskComputation(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    std::vector<uchar>* inlier_mask) const {
  CHECK_NOTNULL(inlier_mask);

  static constexpr double max_dist_from_epi_line_in_px = 3.0;
  static constexpr double confidence_prob = 0.99;
  try {
    findFundamentalMat(pts1, pts2, CV_FM_RANSAC, max_dist_from_epi_line_in_px,
                       confidence_prob, *inlier_mask);
  } catch (...) {
    ROS_WARN(
        "Inlier Mask could not be computed, this can happen if there"
        "are not enough features tracked.");
  }
}
