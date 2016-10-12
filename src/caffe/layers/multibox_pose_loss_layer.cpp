#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multibox_pose_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
void MultiBoxPoseLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // TODO change this
  if (this->layer_param_.propagate_down_size() == 0) {
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    this->layer_param_.add_propagate_down(false);
  }
  const MultiBoxPoseLossParameter& multibox_pose_loss_param =
      this->layer_param_.multibox_pose_loss_param();

  num_ = bottom[0]->num();
  num_priors_ = bottom[4]->height() / 4;

  // Get other parameters.
  CHECK(multibox_pose_loss_param.has_num_classes()) << "Must provide num_classes.";
  num_classes_ = multibox_pose_loss_param.num_classes();
  CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";

  CHECK(multibox_pose_loss_param.has_num_poses()) << "Must provide num_poses.";
  num_poses_ = multibox_pose_loss_param.num_poses();
  CHECK_GE(num_poses_, 1) << "num_poses should not be less than 1.";

  share_location_ = multibox_pose_loss_param.share_location();
  loc_classes_ = share_location_ ? 1 : num_classes_;
    
  share_pose_ = multibox_pose_loss_param.share_pose();
  pose_classes_ = share_pose_ ? 1 : num_poses_; 

  match_type_ = multibox_pose_loss_param.match_type();
  overlap_threshold_ = multibox_pose_loss_param.overlap_threshold();
  use_prior_for_matching_ = multibox_pose_loss_param.use_prior_for_matching();
  background_label_id_ = multibox_pose_loss_param.background_label_id();
  use_difficult_gt_ = multibox_pose_loss_param.use_difficult_gt();
  do_neg_mining_ = multibox_pose_loss_param.do_neg_mining();
  neg_pos_ratio_ = multibox_pose_loss_param.neg_pos_ratio();
  neg_overlap_ = multibox_pose_loss_param.neg_overlap();
  code_type_ = multibox_pose_loss_param.code_type();
  encode_variance_in_target_ = multibox_pose_loss_param.encode_variance_in_target();
  map_object_to_agnostic_ = multibox_pose_loss_param.map_object_to_agnostic();
  if (map_object_to_agnostic_) {
    if (background_label_id_ >= 0) {
      CHECK_EQ(num_classes_, 2);
    } else {
      CHECK_EQ(num_classes_, 1);
    }
  }

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  if (do_neg_mining_) {
    CHECK(share_location_)
        << "Currently only support negative mining if share_location is true.";
    CHECK_GT(neg_pos_ratio_, 0);
  }

  vector<int> loss_shape(1, 1);
  // Set up localization loss layer.
  loc_weight_ = multibox_pose_loss_param.loc_weight();
  loc_loss_type_ = multibox_pose_loss_param.loc_loss_type();
  // fake shape.
  vector<int> loc_shape(1, 1);
  loc_shape.push_back(4);
  loc_pred_.Reshape(loc_shape);
  loc_gt_.Reshape(loc_shape);
  loc_bottom_vec_.push_back(&loc_pred_);
  loc_bottom_vec_.push_back(&loc_gt_);
  loc_loss_.Reshape(loss_shape);
  loc_top_vec_.push_back(&loc_loss_);
  if (loc_loss_type_ == LocLossType_L2) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_l2_loc");
    layer_param.set_type("EuclideanLoss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else if (loc_loss_type_ == LocLossType_SMOOTH_L1) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
    layer_param.set_type("SmoothL1Loss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  } else {
    LOG(FATAL) << "Unknown localization loss type.";
  }



  // Set up confidence loss layer.
  conf_loss_type_ = multibox_pose_loss_param.conf_loss_type();
  conf_bottom_vec_.push_back(&conf_pred_);
  conf_bottom_vec_.push_back(&conf_gt_);
  conf_loss_.Reshape(loss_shape);
  conf_top_vec_.push_back(&conf_loss_);
  if (conf_loss_type_ == ConfLossType_SOFTMAX) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
    layer_param.set_type("SoftmaxWithLoss");
//    layer_param.add_loss_weight(Dtype(1.));
    layer_param.add_loss_weight(Dtype(0.5));
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> conf_shape(1, 1);
    conf_gt_.Reshape(conf_shape);
    conf_shape.push_back(num_classes_);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  } else if (conf_loss_type_ == ConfLossType_LOGISTIC) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_logistic_conf");
    layer_param.set_type("SigmoidCrossEntropyLoss");
    layer_param.add_loss_weight(Dtype(1.));
    // Fake reshape.
    vector<int> conf_shape(1, 1);
    conf_shape.push_back(num_classes_);
    conf_gt_.Reshape(conf_shape);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  } else {
    LOG(FATAL) << "Unknown confidence loss type.";
  }


  // Set up discrete pose classification layer
  // Assume pose loss layer use softmax + cross_entropy
  pose_bottom_vec_.push_back(&pose_pred_);
  pose_bottom_vec_.push_back(&pose_gt_);
  pose_loss_.Reshape(loss_shape);
  pose_top_vec_.push_back(&pose_loss_);
  pose_weight_ = multibox_pose_loss_param.pose_weight();
  if (1) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_pose");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(pose_weight_);
    layer_param.mutable_loss_param()->set_normalization(
        LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> pose_shape(1, 1);
    pose_gt_.Reshape(pose_shape);
    pose_shape.push_back(num_poses_);
    pose_pred_.Reshape(pose_shape);
    pose_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    pose_loss_layer_->SetUp(pose_bottom_vec_, pose_top_vec_);
  }


  // Set up pose regression loss layer.
  pose_reg_weight_ = multibox_pose_loss_param.pose_reg_weight();
  pose_reg_loss_type_ = multibox_pose_loss_param.pose_reg_loss_type();
  // fake shape.
  vector<int> pose_reg_shape(1, 1);
  pose_reg_shape.push_back(4);
  pose_reg_pred_.Reshape(pose_reg_shape);
  pose_reg_gt_.Reshape(pose_reg_shape);
  
  pose_reg_bottom_vec_.push_back(&pose_reg_pred_);
  pose_reg_bottom_vec_.push_back(&pose_reg_gt_);

  pose_reg_loss_.Reshape(pose_reg_shape);
  pose_reg_top_vec_.push_back(&pose_reg_loss_);

  if (loc_loss_type_ == LocLossType_L2) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_l2_loc");
    layer_param.set_type("EuclideanLoss");
    layer_param.add_loss_weight(pose_reg_weight_);
    pose_reg_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    pose_reg_loss_layer_->SetUp(pose_reg_bottom_vec_, pose_reg_top_vec_);
  } else if (loc_loss_type_ == LocLossType_SMOOTH_L1) {
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
    layer_param.set_type("SmoothL1Loss");
    layer_param.add_loss_weight(pose_reg_weight_);
    pose_reg_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    pose_reg_loss_layer_->SetUp(pose_reg_bottom_vec_, pose_reg_top_vec_);
  } else {
    LOG(FATAL) << "Unknown localization loss type.";
  }
}

template <typename Dtype>
void MultiBoxPoseLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  num_ = bottom[0]->num();
  // TODO update this
  num_priors_ = bottom[4]->height() / 4;
  num_gt_ = bottom[5]->height();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(num_priors_ * loc_classes_ * 4, bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  //CHECK_EQ(num_priors_ * pose_classes_ * num_poses_, bottom[2]->channels())
  //    << "Number of priors must match number of pose predictions.";
    CHECK_EQ(num_priors_ * num_poses_, bottom[2]->channels())
      << "Number of priors must match number of pose predictions.";
}

template <typename Dtype>
void MultiBoxPoseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // bottom was [loc, conf, pose, priorbox, label]
  // bottom is now [loc, conf, pose_cls, pose_reg, priorbox, label]
  /*
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* pose_data = bottom[2]->cpu_data();
  const Dtype* prior_data = bottom[3]->cpu_data();
  const Dtype* gt_data = bottom[4]->cpu_data();
  */
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* pose_data = bottom[2]->cpu_data();
  const Dtype* pose_reg_data = bottom[3]->cpu_data();
  const Dtype* prior_data = bottom[4]->cpu_data();
  const Dtype* gt_data = bottom[5]->cpu_data();

  // Retrieve all ground truth --done
  GetGroundTruthPose(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                 &all_gt_bboxes_);

  // Retrieve all discrete pose predictions.
  vector<map<int, vector<vector<float> > > > all_pose_preds;
  GetPosePredictions(pose_data, num_, num_poses_, num_priors_, &all_pose_preds);

  // Retrieve all pose regression predictions
  vector<  map<int, vector< vector<float> > > > all_pose_reg_preds;
  GetPoseRegPredictions(pose_reg_data, num_, num_priors_, pose_classes_, share_pose_, 
      &all_pose_reg_preds);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension. --dont change
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Retrieve all predictions. -- dont change. can modify to get regression preds
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num_, num_priors_, loc_classes_, share_location_,
                    &all_loc_preds);

  // Retrieve max scores for each prior. Used in negative mining.
  vector<vector<float> > all_max_scores;
  if (do_neg_mining_) {
    GetMaxConfidenceScores(conf_data, num_, num_priors_, num_classes_,
                           background_label_id_, conf_loss_type_,
                           &all_max_scores);
  }

  num_matches_ = 0;
  int num_negs = 0;
  for (int i = 0; i < num_; ++i) {
    map<int, vector<int> > match_indices;
    vector<int> neg_indices;
    // Check if there is ground truth for current image.
    if (all_gt_bboxes_.find(i) == all_gt_bboxes_.end()) {
      // There is no gt for current image. All predictions are negative.
      all_match_indices_.push_back(match_indices);
      all_neg_indices_.push_back(neg_indices);
      continue;
    }
    // Find match between predictions and ground truth.
    const vector<NormalizedBBox>& gt_bboxes = all_gt_bboxes_.find(i)->second;
    
    vector<int> labels; 
    labels.resize(num_classes_);
    // get the class labels that appear in the gt
    for (int g=0; g < gt_bboxes.size(); g++) {
        labels[gt_bboxes[g].label()] = 1;
    }

    map<int, vector<float> > match_overlaps;
    if (!use_prior_for_matching_) {
      for (int c = 0; c < loc_classes_; ++c) {
        int label = share_location_ ? -1 : c;
        if (!share_location_ && label == background_label_id_) {
          // Ignore background loc predictions.
          continue;
        }
        // Decode the prediction into bbox first.
        vector<NormalizedBBox> loc_bboxes;
        DecodeBBoxes(prior_bboxes, prior_variances,
                     code_type_, encode_variance_in_target_,
                     all_loc_preds[i][label], &loc_bboxes);
        MatchBBox(gt_bboxes, loc_bboxes, label, match_type_,
                  overlap_threshold_, &match_indices[label],
                  &match_overlaps[label]);
      }
    } else {
      // Use prior bboxes to match against all ground truth.
      vector<int> temp_match_indices;
      vector<float> temp_match_overlaps;
      const int label = -1;
      MatchBBox(gt_bboxes, prior_bboxes, label, match_type_, overlap_threshold_,
                &temp_match_indices, &temp_match_overlaps);
      if (share_location_) {
        match_indices[label] = temp_match_indices;
        match_overlaps[label] = temp_match_overlaps;
      } else {
        // Get ground truth label for each ground truth bbox.
        vector<int> gt_labels;
        for (int g = 0; g < gt_bboxes.size(); ++g) {
          gt_labels.push_back(gt_bboxes[g].label());
        }
        // Distribute the matching results to different loc_class.
        for (int c = 0; c < loc_classes_; ++c) {
          if (c == background_label_id_) {
            // Ignore background loc predictions.
            continue;
          }
          match_indices[c].resize(temp_match_indices.size(), -1);
          match_overlaps[c] = temp_match_overlaps;
          for (int m = 0; m < temp_match_indices.size(); ++m) {
            if (temp_match_indices[m] != -1) {
              const int gt_idx = temp_match_indices[m];
              CHECK_LT(gt_idx, gt_labels.size());
              if (c == gt_labels[gt_idx]) {
                match_indices[c][m] = gt_idx;
              }
            }
          }
        }
      }
    }
    // Record matching statistics.
    for (map<int, vector<int> >::iterator it = match_indices.begin();
         it != match_indices.end(); ++it) {
      const int label = it->first;
      // Get positive indices.
      int num_pos = 0;
      for (int m = 0; m < match_indices[label].size(); ++m) {
        if (match_indices[label][m] != -1) {
          ++num_pos;
        }
      }
      num_matches_ += num_pos;
      if (do_neg_mining_) {
        // Get max scores for all the non-matched priors.
        vector<pair<float, int> > scores_indices;
        int num_neg = 0;
        for (int m = 0; m < match_indices[label].size(); ++m) {
          if (match_indices[label][m] == -1 &&
              match_overlaps[label][m] < neg_overlap_) {
            scores_indices.push_back(std::make_pair(all_max_scores[i][m], m));
            ++num_neg;
          }
        }
        // Pick top num_neg negatives.
        num_neg = std::min(static_cast<int>(num_pos * neg_pos_ratio_), num_neg);
        std::sort(scores_indices.begin(), scores_indices.end(),
                  SortScorePairDescend<int>);
        for (int n = 0; n < num_neg; ++n) {
          neg_indices.push_back(scores_indices[n].second);
        }
        num_negs += num_neg;
      }
    }
    all_match_indices_.push_back(match_indices);
    all_neg_indices_.push_back(neg_indices);
  }

  if (num_matches_ >= 1) {
    // Form data to pass on to loc_loss_layer_.
    vector<int> loc_shape(2);
    loc_shape[0] = 1;
    loc_shape[1] = num_matches_ * 4;
    loc_pred_.Reshape(loc_shape);
    loc_gt_.Reshape(loc_shape);
    Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
    Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();


    // RIC adding angle regression loss
    vector<int> pose_reg_shape(2);
    pose_reg_shape[0] = 1;
    pose_reg_shape[1] = num_matches_ * 3;
    pose_reg_pred_.Reshape(pose_reg_shape);
    pose_reg_gt_.Reshape(pose_reg_shape);
    Dtype* pose_reg_pred_data = pose_reg_pred_.mutable_cpu_data();
    Dtype* pose_reg_gt_data = pose_reg_gt_.mutable_cpu_data();


    // Form data to pass on to pose_loss_layer_.
    // Assume pose loss layer is softmax - crossentropy
    // RIC changing to mach loc_shape style above
    // NVM dont change it
    
    vector<int> pose_shape;
    pose_shape.push_back(num_matches_); 
    pose_gt_.Reshape(pose_shape);
    pose_shape.push_back(num_poses_); 
    pose_pred_.Reshape(pose_shape);
    Dtype* pose_pred_data = pose_pred_.mutable_cpu_data();
    Dtype* pose_gt_data = pose_gt_.mutable_cpu_data();
    //vector<int> pose_shape(2);
    //pose_shape[0] = num_poses_; 
    //pose_shape[1] = num_matches_; 
    //pose_gt_.Reshape(pose_shape);
    //pose_pred_.Reshape(pose_shape);
    //Dtype* pose_pred_data = pose_pred_.mutable_cpu_data();
    //Dtype* pose_gt_data = pose_gt_.mutable_cpu_data();


    int count = 0;
    for (int i = 0; i < num_; ++i) {
      for (map<int, vector<int> >::iterator it = all_match_indices_[i].begin();
           it != all_match_indices_[i].end(); ++it) {
        
        const int label = it->first;
        const vector<int>& match_index = it->second;
        
        CHECK(all_loc_preds[i].find(label) != all_loc_preds[i].end());

        const vector<NormalizedBBox>& loc_pred = all_loc_preds[i][label];

        const vector<vector<float> >& poses = all_pose_preds[i].find(-1)->second;
        map<int, vector<vector<float> > >& pose_reg_preds = all_pose_reg_preds[i]; 

        for (int j = 0; j < match_index.size(); ++j) {
          if (match_index[j] == -1) {
            continue;
          }
          // Store location prediction.
          CHECK_LT(j, loc_pred.size());
          loc_pred_data[count * 4] = loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] = loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] = loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] = loc_pred[j].ymax();

          // Store encoded ground truth.
          const int gt_idx = match_index[j];
          
          CHECK(all_gt_bboxes_.find(i) != all_gt_bboxes_.end());
          CHECK_LT(gt_idx, all_gt_bboxes_[i].size());

          const NormalizedBBox& gt_bbox = all_gt_bboxes_[i][gt_idx];
          NormalizedBBox gt_encode;
          CHECK_LT(j, prior_bboxes.size());
          EncodeBBox(prior_bboxes[j], prior_variances[j], code_type_,
                     encode_variance_in_target_, gt_bbox, &gt_encode);
          loc_gt_data[count * 4] = gt_encode.xmin();
          loc_gt_data[count * 4 + 1] = gt_encode.ymin();
          loc_gt_data[count * 4 + 2] = gt_encode.xmax();
          loc_gt_data[count * 4 + 3] = gt_encode.ymax();
          if (encode_variance_in_target_) {
            for (int k = 0; k < 4; ++k) {
              CHECK_GT(prior_variances[j][k], 0);
              loc_pred_data[count * 4 + k] /= prior_variances[j][k];
              loc_gt_data[count * 4 + k] /= prior_variances[j][k];
            }
          }

          // Copy pose predictions
          for (int k = 0; k < num_poses_; ++k) {
            pose_pred_data[count * num_poses_ + k] = poses[j][k];
          }

          // Get pose regression predictions
          int pose_bin = share_pose_ ? -1 : gt_bbox.pose();
          vector<vector<float> >& pose_reg = pose_reg_preds.find(pose_bin)->second;
          vector<float> pose_reg_out = pose_reg[j];
          pose_reg_pred_data[count*3] = pose_reg_out[0];
          pose_reg_pred_data[count*3 + 1] = pose_reg_out[1];
          pose_reg_pred_data[count*3 + 2] = pose_reg_out[2]; 

          
          // Store pose ground truth.
          pose_gt_data[count] = gt_bbox.pose();
          //const int poseLabel = gt_bbox.pose();
          pose_reg_gt_data[count * 3] = gt_bbox.eone();
          pose_reg_gt_data[count * 3 + 1] = gt_bbox.etwo();
          pose_reg_gt_data[count * 3 + 2] = gt_bbox.ethree();

          
          // We are going to use same pose for each class
          // share pose will control the regression targets
          //const int label_off = share_pose_? 0 : ricLabel;
          //int offset = num_poses_ * pose_classes_;

          //LOG(INFO) << "label off " << label_off;
          //LOG(INFO) << "offset is " << offset;
          //LOG(INFO) << "gt pose is " << gt_bbox.azilabel();
          //LOG(INFO) << "ric label is " << ricLabel;
          //LOG(INFO) << "count is " << count;
          //std::cin.ignore();
          /*

          caffe_copy<Dtype>(num_poses_, pose_data + j * num_poses_,
                                pose_pred_data + count * num_poses_);

          const int label_off_reg = share_pose_? 0 : poseLabel;
          const int offset = num_poses_ * 3; // euler angles

          caffe_copy<Dtype>(3, pose_reg_data + j * offset + label_off_reg*3,
                                pose_reg_pred_data + count * 3);
          */

          ++count;
        }
      }
      //pose_data += num_priors_* num_poses_ * pose_classes_;
      //pose_data += num_priors_* num_poses_; 
      //pose_gt_data += num_priors_ * 3 * pose_classes_;
    }
    loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
    loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);

    pose_loss_layer_->Reshape(pose_bottom_vec_, pose_top_vec_);
    pose_loss_layer_->Forward(pose_bottom_vec_, pose_top_vec_);

    pose_reg_loss_layer_->Reshape(pose_reg_bottom_vec_, pose_reg_top_vec_);
    pose_reg_loss_layer_->Forward(pose_reg_bottom_vec_, pose_reg_top_vec_);
  }

  // Form data to pass on to conf_loss_layer_.
  if (do_neg_mining_) {
    num_conf_ = num_matches_ + num_negs;
  } else {
    num_conf_ = num_ * num_priors_;
  }
  if (num_conf_ >= 1) {
    // Reshape the confidence data.
    vector<int> conf_shape;
    if (conf_loss_type_ == ConfLossType_SOFTMAX) {
      conf_shape.push_back(num_conf_);
      conf_gt_.Reshape(conf_shape);
      conf_shape.push_back(num_classes_);
      conf_pred_.Reshape(conf_shape);
    } else if (conf_loss_type_ == ConfLossType_LOGISTIC) {
      conf_shape.push_back(1);
      conf_shape.push_back(num_conf_);
      conf_shape.push_back(num_classes_);
      conf_gt_.Reshape(conf_shape);
      conf_pred_.Reshape(conf_shape);
    } else {
      LOG(FATAL) << "Unknown confidence loss type.";
    }
    if (!do_neg_mining_) {
      // Consider all scores.
      // Share data and diff with bottom[1].
      CHECK_EQ(conf_pred_.count(), bottom[1]->count());
      conf_pred_.ShareData(*(bottom[1]));
    }
    Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
    Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
    caffe_set(conf_gt_.count(), Dtype(background_label_id_), conf_gt_data);
    int count = 0;
    for (int i = 0; i < num_; ++i) {
      if (all_gt_bboxes_.find(i) != all_gt_bboxes_.end()) {
        // Save matched (positive) bboxes scores and labels.
        const map<int, vector<int> >& match_indices = all_match_indices_[i];
        for (int j = 0; j < num_priors_; ++j) {
          for (map<int, vector<int> >::const_iterator it =
               match_indices.begin(); it != match_indices.end(); ++it) {
            const vector<int>& match_index = it->second;
            CHECK_EQ(match_index.size(), num_priors_);
            if (match_index[j] == -1) {
              continue;
            }
            const int gt_label = map_object_to_agnostic_ ?
                background_label_id_ + 1 :
                all_gt_bboxes_[i][match_index[j]].label();
            int idx = do_neg_mining_ ? count : j;
            switch (conf_loss_type_) {
              case ConfLossType_SOFTMAX:
                conf_gt_data[idx] = gt_label;
                break;
              case ConfLossType_LOGISTIC:
                conf_gt_data[idx * num_classes_ + gt_label] = 1;
                break;
              default:
                LOG(FATAL) << "Unknown conf loss type.";
            }
            if (do_neg_mining_) {
              // Copy scores for matched bboxes.
              caffe_copy<Dtype>(num_classes_, conf_data + j * num_classes_,
                                conf_pred_data + count * num_classes_);
              ++count;
            }
          }
        }
        if (do_neg_mining_) {
          // Save negative bboxes scores and labels.
          for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
            int j = all_neg_indices_[i][n];
            CHECK_LT(j, num_priors_);
            caffe_copy<Dtype>(num_classes_, conf_data + j * num_classes_,
                              conf_pred_data + count * num_classes_);
            switch (conf_loss_type_) {
              case ConfLossType_SOFTMAX:
                conf_gt_data[count] = background_label_id_;
                break;
              case ConfLossType_LOGISTIC:
                conf_gt_data[count * num_classes_ + background_label_id_] = 1;
                break;
              default:
                LOG(FATAL) << "Unknown conf loss type.";
            }
            ++count;
          }
        }
      }
      // Go to next image.
      if (do_neg_mining_) {
        conf_data += bottom[1]->offset(1);
      } else {
        conf_gt_data += num_priors_;
      }
    }
    conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
    conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);
  }

  top[0]->mutable_cpu_data()[0] = 0;
  if (this->layer_param_.propagate_down(0)) {
    // TODO(weiliu89): Understand why it needs to divide 2.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_matches_);
    Dtype temp = loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
    top[0]->mutable_cpu_data()[0] += temp;
    LOG(INFO) << "Multibox Loss ( Loc loss ): " << temp;
  }
  if (this->layer_param_.propagate_down(1)) {
    // TODO(weiliu89): Understand why it needs to divide 2.
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_matches_);
    Dtype temp = conf_loss_.cpu_data()[0] / normalizer;
    top[0]->mutable_cpu_data()[0] += temp;
    LOG(INFO) << "Multibox Loss ( Conf loss ): " << temp;
  }
  if (this->layer_param_.propagate_down(2)) {
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_matches_);
    Dtype temp = pose_weight_ * pose_loss_.cpu_data()[0] / normalizer;
    top[0]->mutable_cpu_data()[0] += temp;
    LOG(INFO) << "Multibox Loss ( Pose loss ): " << temp;
  }

    if (this->layer_param_.propagate_down(3)) {
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
        normalization_, num_, num_priors_, num_matches_);
    Dtype temp = pose_reg_weight_ * pose_reg_loss_.cpu_data()[0] / normalizer;
    top[0]->mutable_cpu_data()[0] += temp;
    LOG(INFO) << "Multibox Loss ( Pose Regression loss ): " << temp;
  }
   
}

template <typename Dtype>
void MultiBoxPoseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[4]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to prior inputs.";
  }
  if (propagate_down[5]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
  }

  // Back propagate on location prediction.
  if (propagate_down[0]) {
    Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
    if (num_matches_ >= 1) {
      vector<bool> loc_propagate_down;
      // Only back propagate on prediction, not ground truth.
      loc_propagate_down.push_back(true);
      loc_propagate_down.push_back(false);
      loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
                                loc_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[0].
      const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
      int count = 0;
      for (int i = 0; i < num_; ++i) {
        for (map<int, vector<int> >::iterator it =
             all_match_indices_[i].begin();
             it != all_match_indices_[i].end(); ++it) {
          const int label = share_location_ ? 0 : it->first;
          const vector<int>& match_index = it->second;
          for (int j = 0; j < match_index.size(); ++j) {
            if (match_index[j] == -1) {
              continue;
            }
            // Copy the diff to the right place.
            int start_idx = loc_classes_ * 4 * j + label * 4;
            caffe_copy<Dtype>(4, loc_pred_diff + count * 4,
                              loc_bottom_diff + start_idx);
            ++count;
          }
        }
        loc_bottom_diff += bottom[0]->offset(1);
      }
    }
  }

  // Back propagate on confidence prediction.
  if (propagate_down[1]) {
    Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
    caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);
    if (num_conf_ >= 1) {
      vector<bool> conf_propagate_down;
      // Only back propagate on prediction, not ground truth.
      conf_propagate_down.push_back(true);
      conf_propagate_down.push_back(false);
      conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
                                 conf_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(conf_pred_.count(), loss_weight,
                 conf_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[1].
      const Dtype* conf_pred_diff = conf_pred_.cpu_diff();
      if (do_neg_mining_) {
        int count = 0;
        for (int i = 0; i < num_; ++i) {
          // Copy matched (positive) bboxes scores' diff.
          const map<int, vector<int> >& match_indices = all_match_indices_[i];
          for (int j = 0; j < num_priors_; ++j) {
            for (map<int, vector<int> >::const_iterator it =
                 match_indices.begin(); it != match_indices.end(); ++it) {
              const vector<int>& match_index = it->second;
              CHECK_EQ(match_index.size(), num_priors_);
              if (match_index[j] == -1) {
                continue;
              }
              // Copy the diff to the right place.
              caffe_copy<Dtype>(num_classes_,
                                conf_pred_diff + count * num_classes_,
                                conf_bottom_diff + j * num_classes_);
              ++count;
            }
          }
          // Copy negative bboxes scores' diff.
          for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
            int j = all_neg_indices_[i][n];
            CHECK_LT(j, num_priors_);
            caffe_copy<Dtype>(num_classes_,
                              conf_pred_diff + count * num_classes_,
                              conf_bottom_diff + j * num_classes_);
            ++count;
          }
          conf_bottom_diff += bottom[1]->offset(1);
        }
      } else {
        // The diff is already computed and stored.
        bottom[1]->ShareDiff(conf_pred_);
      }
    }
  }

  // Back propagate on pose prediction.
  if (propagate_down[2]) {
    Dtype* pose_bottom_diff = bottom[2]->mutable_cpu_diff();
    caffe_set(bottom[2]->count(), Dtype(0), pose_bottom_diff);
    if (num_matches_ >= 1) {
      vector<bool> pose_propagate_down;
      // Only back propagate on prediction, not ground truth.
      pose_propagate_down.push_back(true);
      pose_propagate_down.push_back(false);
      pose_loss_layer_->Backward(pose_top_vec_, pose_propagate_down,
                                pose_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(pose_pred_.count(), loss_weight, pose_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[0].
      const Dtype* pose_pred_diff = pose_pred_.cpu_diff();

      int count = 0;
      for (int i = 0; i < num_; ++i) {
        for (map<int, vector<int> >::iterator it =
             all_match_indices_[i].begin();
             it != all_match_indices_[i].end(); ++it) {
          
          const vector<int>& match_index = it->second;
          for (int j = 0; j < match_index.size(); ++j) {
            if (match_index[j] == -1) {
              continue;
            }

            //const int gt_idx = match_index[j];
            //const int ricLabel = all_gt_bboxes_[i][gt_idx].label();
            //const int label = share_pose_ ? 0 : ricLabel;

            // Copy the diff to the right place.
            //int start_idx = pose_classes_ * num_poses_ * j + label * num_poses_;
            //caffe_copy<Dtype>(num_poses_, pose_pred_diff + count * num_poses_,
            //                  pose_bottom_diff + start_idx);
            int start_idx = num_poses_ * j;
            caffe_copy<Dtype>(num_poses_, pose_pred_diff + count * num_poses_,
                              pose_bottom_diff + start_idx);

            ++count;
          }
        }
        pose_bottom_diff += bottom[2]->offset(1);
      }
    }
  }

    // Back propagate on pose regression
  if (propagate_down[3]) {
    Dtype* pose_reg_bottom_diff = bottom[3]->mutable_cpu_diff();
    caffe_set(bottom[3]->count(), Dtype(0), pose_reg_bottom_diff);
    if (num_matches_ >= 1) {
      vector<bool> pose_reg_propagate_down;
      // Only back propagate on prediction, not ground truth.
      pose_reg_propagate_down.push_back(true);
      pose_reg_propagate_down.push_back(false);
      pose_reg_loss_layer_->Backward(pose_reg_top_vec_, pose_reg_propagate_down,
                                pose_reg_bottom_vec_);
      // Scale gradient.
      Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
          normalization_, num_, num_priors_, num_matches_);
      Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
      caffe_scal(pose_reg_pred_.count(), loss_weight, pose_reg_pred_.mutable_cpu_diff());
      // Copy gradient back to bottom[0].
      const Dtype* pose_reg_pred_diff = pose_reg_pred_.cpu_diff();
      int count = 0;
      for (int i = 0; i < num_; ++i) {
        for (map<int, vector<int> >::iterator it =
             all_match_indices_[i].begin();
             it != all_match_indices_[i].end(); ++it) {

          //const int label = share_location_ ? 0 : it->first;
          const vector<int>& match_index = it->second;
          for (int j = 0; j < match_index.size(); ++j) {
            if (match_index[j] == -1) {
              continue;
            }

            const int gt_idx = match_index[j];
            const int gt_pose = all_gt_bboxes_[i][gt_idx].pose();

            const int diff_label_off = share_pose_ ? 0 : gt_pose;

            // Copy the diff to the right place.
            int start_idx = pose_classes_ * 3 * j + diff_label_off * 3;
            caffe_copy<Dtype>(3, pose_reg_pred_diff + count * 3,
                              pose_reg_bottom_diff + start_idx);
            ++count;
          }
        }
        pose_reg_bottom_diff += bottom[3]->offset(1);
      }
    }
  }

  // After backward, remove match statistics.
  all_match_indices_.clear();
  all_neg_indices_.clear();
}


#ifdef CPU_ONLY
STUB_GPU(MultiBoxPoseLossLayer);
#endif

INSTANTIATE_CLASS(MultiBoxPoseLossLayer);
REGISTER_LAYER_CLASS(MultiBoxPoseLoss);

}  // namespace caffe
