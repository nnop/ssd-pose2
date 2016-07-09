#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_pose_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionPoseOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* loc_data = bottom[0]->gpu_data();
  const Dtype* conf_data = bottom[1]->gpu_data();
  const Dtype* pose_data = bottom[2]->cpu_data();
  //const Dtype* pose_data = bottom[2]->gpu_data();
  const Dtype* prior_data = bottom[3]->gpu_data();
  const int num = bottom[0]->num();

  //LOG(INFO) << "pose data num " << bottom[2]->num();
  //LOG(INFO) << "pose data channels " << bottom[2]->channels();
  //LOG(INFO) << "pose data height " << bottom[2]->height();
  //LOG(INFO) << "pose data width " << bottom[2]->width();
  //LOG(INFO) << "pose data count " << bottom[2]->count();
  //LOG(INFO) << "num poses " << num_poses_;
  //LOG(INFO) << "num priors " << num_priors_;
  //LOG(INFO) << "num pose classes " << num_pose_classes_;


  //map<int, map<int, vector<vector<float> > > > all_pose_preds;
  vector< map<int, vector<vector<float> > > > all_pose_preds;
  GetPosePredictions(pose_data, num, num_poses_, num_priors_, num_pose_classes_, 
                    share_pose_, &all_pose_preds);
  
  //const int pose_count = pose_data.count();
  //GetPoseGPU<Dtype>(pose_count, pose_data, num, num_poses_, num_priors_, num_pose_classes_, 
  //  share_pose_, &all_pose_preds);


  // Decode predictions.
  Blob<Dtype> bbox_preds;
  bbox_preds.ReshapeLike(*(bottom[0]));
  Dtype* bbox_data = bbox_preds.mutable_gpu_data();
  const int loc_count = bbox_preds.count();
  DecodeBBoxesGPU<Dtype>(loc_count, loc_data, prior_data, code_type_,
      variance_encoded_in_target_, num_priors_, share_location_,
      num_loc_classes_, background_label_id_, bbox_data);



  if (!share_location_) {
    Blob<Dtype> bbox_permute;
    bbox_permute.ReshapeLike(*(bottom[0]));
    Dtype* bbox_permute_data = bbox_permute.mutable_gpu_data();
    PermuteDataGPU<Dtype>(loc_count, bbox_data, num_loc_classes_, num_priors_,
        4, bbox_permute_data);
    caffe_copy(loc_count, bbox_permute_data, bbox_data);
  }

  // Retrieve all confidences.
  const Dtype* conf_cpu_data;
  Blob<Dtype> conf_permute;
  conf_permute.ReshapeLike(*(bottom[1]));
  Dtype* conf_permute_data = conf_permute.mutable_gpu_data();
  PermuteDataGPU<Dtype>(conf_permute.count(), conf_data,
      num_classes_, num_priors_, 1, conf_permute_data);
  conf_cpu_data = conf_permute.cpu_data();



  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  // spot b
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > indices;
    int num_det = 0;
    const int start_idx = i * num_classes_ * num_priors_;
    for (int c = 0; c < num_classes_; ++c) {
      if (c != background_label_id_) {
        ApplyNMSGPU(bbox_data, conf_cpu_data + start_idx + c * num_priors_,
            num_priors_, confidence_threshold_, top_k_, nms_threshold_,
            &(indices[c]));
        num_det += indices[c].size();
      }
      if (!share_location_) {
        bbox_data += num_priors_ * 4;
      }
    }

    if (share_location_) {
      bbox_data += num_priors_ * 4;
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          float score = conf_cpu_data[start_idx + label * num_priors_ + idx];
          score_index_pairs.push_back(std::make_pair(
                  score, std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }

  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    return;
  }

  
  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(9);
  top[0]->Reshape(top_shape);
  Dtype* top_data = top[0]->mutable_cpu_data();


  const Dtype* bbox_cpu_data = bbox_preds.cpu_data();

  boost::filesystem::path output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {
    //map<int, vector<vector<float> > >& pose_preds = all_pose_preds.find(i)->second;
    map<int, vector<vector<float> > >& pose_preds = all_pose_preds[i];
    int start_idx = i * num_classes_ * num_priors_;
    for (int c = 0; c < num_classes_; ++c) {
      if (all_indices[i].find(c) == all_indices[i].end()) {
        if (!share_location_) {
          bbox_cpu_data += num_priors_ * 4;
        }
        continue;
      }

      vector<int>& indices = all_indices[i].find(c)->second;

      int label = c;

      int pose_label = share_pose_ ? -1 : label;
      if (pose_preds.find(pose_label) == pose_preds.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find Pose predictions for " << pose_label;
        continue;
      }
      const vector<vector<float> >& poses = pose_preds.find(pose_label)->second;
      
      // Retrieve detection data.
      bool clip_bbox = true;
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[j * 9] = i;
        top_data[j * 9 + 1] = label;
        top_data[j * 9 + 2] = conf_cpu_data[start_idx + label * num_priors_ + idx];
        if (clip_bbox) {
          for (int k = 0; k < 4; ++k) {
          top_data[j * 9 + 3 + k] = std::max(
                std::min(bbox_cpu_data[idx * 4 + k], Dtype(1)), Dtype(0));
          }
        } else {
          for (int k = 0; k < 4; ++k) {
            top_data[j * 9 + 3 + k] = bbox_cpu_data[idx * 4 + k];
          }
        }

        vector<float> target_pose= poses[idx];
        vector<float>::iterator result;
        result = max_element(target_pose.begin(), target_pose.end());
        // fix this 
        top_data[j * 9 + 7] = distance(target_pose.begin(), result);
        top_data[j * 9 + 8] = *(result);

      }
      if (!share_location_) {
        bbox_cpu_data += num_priors_ * 4;
      }
      if (indices.size() == 0) {
        continue;
      }
      if (need_save_) {
        CHECK(label_to_name_.find(c) != label_to_name_.end())
            << "Cannot find label: " << c << " in the label map.";
        CHECK_LT(name_count_, names_.size());
        int height = sizes_[name_count_].first;
        int width = sizes_[name_count_].second;
        for (int j = 0; j < indices.size(); ++j) {
          float score = top_data[j * 7 + 2];
          float xmin = top_data[j * 7 + 3] * width;
          float ymin = top_data[j * 7 + 4] * height;
          float xmax = top_data[j * 7 + 5] * width;
          float ymax = top_data[j * 7 + 6] * height;
          ptree pt_xmin, pt_ymin, pt_width, pt_height;
          pt_xmin.put<float>("", round(xmin * 100) / 100.);
          pt_ymin.put<float>("", round(ymin * 100) / 100.);
          pt_width.put<float>("", round((xmax - xmin) * 100) / 100.);
          pt_height.put<float>("", round((ymax - ymin) * 100) / 100.);

          ptree cur_bbox;
          cur_bbox.push_back(std::make_pair("", pt_xmin));
          cur_bbox.push_back(std::make_pair("", pt_ymin));
          cur_bbox.push_back(std::make_pair("", pt_width));
          cur_bbox.push_back(std::make_pair("", pt_height));

          ptree cur_det;
          cur_det.put("image_id", names_[name_count_]);
          if (output_format_ == "ILSVRC") {
            cur_det.put<int>("category_id", c);
          } else {
            cur_det.put("category_id", label_to_name_[c].c_str());
          }
          cur_det.add_child("bbox", cur_bbox);
          cur_det.put<float>("score", score);

          detections_.push_back(std::make_pair("", cur_det));
        }
      }
      top_data += indices.size() * 9;
    }
    if (share_location_) {
      bbox_cpu_data += num_priors_ * 4;
    }
    if (need_save_) {
      ++name_count_;
      if (name_count_ % num_test_image_ == 0) {
        if (output_format_ == "VOC") {
          map<string, std::ofstream*> outfiles;
          for (int c = 0; c < num_classes_; ++c) {
            if (c == background_label_id_) {
              continue;
            }
            string label_name = label_to_name_[c];
            boost::filesystem::path file(
                output_name_prefix_ + label_name + ".txt");
            boost::filesystem::path out_file = output_directory / file;
            outfiles[label_name] = new std::ofstream(out_file.string().c_str(),
                std::ofstream::out);
          }
          BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
            ptree pt = det.second;
            string label_name = pt.get<string>("category_id");
            if (outfiles.find(label_name) == outfiles.end()) {
              std::cout << "Cannot find " << label_name << std::endl;
              continue;
            }
            string image_name = pt.get<string>("image_id");
            float score = pt.get<float>("score");
            vector<int> bbox;
            BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
              bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
            }
            *(outfiles[label_name]) << image_name;
            *(outfiles[label_name]) << " " << score;
            *(outfiles[label_name]) << " " << bbox[0] << " " << bbox[1];
            *(outfiles[label_name]) << " " << bbox[0] + bbox[2];
            *(outfiles[label_name]) << " " << bbox[1] + bbox[3];
            *(outfiles[label_name]) << std::endl;
          }
          for (int c = 0; c < num_classes_; ++c) {
            if (c == background_label_id_) {
              continue;
            }
            string label_name = label_to_name_[c];
            outfiles[label_name]->flush();
            outfiles[label_name]->close();
            delete outfiles[label_name];
          }
        } else if (output_format_ == "COCO") {
          boost::filesystem::path output_directory(output_directory_);
          boost::filesystem::path file(output_name_prefix_ + ".json");
          boost::filesystem::path out_file = output_directory / file;
          std::ofstream outfile;
          outfile.open(out_file.string().c_str(), std::ofstream::out);

          boost::regex exp("\"(null|true|false|-?[0-9]+(\\.[0-9]+)?)\"");
          ptree output;
          output.add_child("detections", detections_);
          std::stringstream ss;
          write_json(ss, output);
          std::string rv = boost::regex_replace(ss.str(), exp, "$1");
          outfile << rv.substr(rv.find("["), rv.rfind("]") - rv.find("["))
              << std::endl << "]" << std::endl;
        } else if (output_format_ == "ILSVRC") {
          boost::filesystem::path output_directory(output_directory_);
          boost::filesystem::path file(output_name_prefix_ + ".txt");
          boost::filesystem::path out_file = output_directory / file;
          std::ofstream outfile;
          outfile.open(out_file.string().c_str(), std::ofstream::out);

          BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
            ptree pt = det.second;
            int label = pt.get<int>("category_id");
            string image_name = pt.get<string>("image_id");
            float score = pt.get<float>("score");
            vector<int> bbox;
            BOOST_FOREACH(ptree::value_type &elem, pt.get_child("bbox")) {
              bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
            }
            outfile << image_name << " " << label << " " << score;
            outfile << " " << bbox[0] << " " << bbox[1];
            outfile << " " << bbox[0] + bbox[2];
            outfile << " " << bbox[1] + bbox[3];
            outfile << std::endl;
          }
        }
        name_count_ = 0;
      }
    }
  }
  if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs;
    this->data_transformer_->TransformInv(bottom[3], &cv_imgs);
    vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
    VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
        label_to_display_name_);
#endif  // USE_OPENCV
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(DetectionPoseOutputLayer);

} // namespace caffe

