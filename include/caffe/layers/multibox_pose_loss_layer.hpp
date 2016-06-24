#ifndef CAFFE_MULTIBOX_POSE_LOSS_LAYER_HPP_
#define CAFFE_MULTIBOX_POSE_LOSS_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Perform MultiBox operations. Including the following:
 *
 *  - decode the predictions.
 *  - perform matching between priors/predictions and ground truth.
 *  - use matched boxes and confidences to compute loss.
 *
 */
template <typename Dtype>
class MultiBoxPoseLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiBoxPoseLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiBoxPoseLoss"; }
  // bottom[0] stores the location predictions.
  // bottom[0] stores the confidence predictions.
  // bottom[2] stores the prior bounding boxes.
  // bottom[3] stores the ground truth bounding boxes.
  virtual inline int ExactNumBottomBlobs() const { return 5; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // The internal localization loss layer.
  shared_ptr<Layer<Dtype> > loc_loss_layer_;
  LocLossType loc_loss_type_;
  float loc_weight_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> loc_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> loc_top_vec_;
  // blob which stores the matched location prediction.
  Blob<Dtype> loc_pred_;
  // blob which stores the corresponding matched ground truth.
  Blob<Dtype> loc_gt_;
  // localization loss.
  Blob<Dtype> loc_loss_;

  // The internal confidence loss layer.
  shared_ptr<Layer<Dtype> > conf_loss_layer_;
  ConfLossType conf_loss_type_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> conf_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> conf_top_vec_;
  // blob which stores the confidence prediction.
  Blob<Dtype> conf_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> conf_gt_;
  // confidence loss.
  Blob<Dtype> conf_loss_;

  // The internal pose loss layer.
  shared_ptr<Layer<Dtype> > pose_loss_layer_;
  // bottom vector holder used in Forward function.
  vector<Blob<Dtype>*> pose_bottom_vec_;
  // top vector holder used in Forward function.
  vector<Blob<Dtype>*> pose_top_vec_;
  // blob which stores the pose prediction.
  Blob<Dtype> pose_pred_;
  // blob which stores the corresponding ground truth label.
  Blob<Dtype> pose_gt_;
  // pose loss.
  Blob<Dtype> pose_loss_;





  int num_classes_;
  int num_poses_;
  bool share_location_;
  bool share_pose_;
  MatchType match_type_;
  float overlap_threshold_;
  bool use_prior_for_matching_;
  int background_label_id_;
  bool use_difficult_gt_;
  bool do_neg_mining_;
  float neg_pos_ratio_;
  float neg_overlap_;
  CodeType code_type_;
  bool encode_variance_in_target_;
  bool map_object_to_agnostic_;

  int loc_classes_;
  int pose_classes_;
  int num_gt_;
  int num_;
  int num_priors_;

  int num_matches_;
  int num_conf_;
  vector<map<int, vector<int> > > all_match_indices_;
  vector<vector<int> > all_neg_indices_;
  map<int, vector<NormalizedBBox> > all_gt_bboxes;

  // How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
