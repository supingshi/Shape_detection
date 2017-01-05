#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl_ros/point_cloud.h>
#include <vector>
#include <math.h>
#include <map>

// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
// #include <pcl/io/pcdio.h>

#include <tf/transform_listener.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/String.h>
#include <pcl/impl/point_types.hpp>
#include <sstream>
#include <iostream>
#include <string>
#include <ros/package.h>
#include <std_msgs/Int8.h>
#include <custom_msgs/Object.h>
#include <custom_msgs/color_center.h>
#include <ras_msgs/RAS_Evidence.h>
#include <sensor_msgs/Image.h>
#include <stdio.h>

#include <pcl/point_types_conversion.h>


// Action lib stuff
#include <vision/ClassifyAction.h>
#include <actionlib/server/simple_action_server.h>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

typedef actionlib::SimpleActionServer<vision::ClassifyAction> ClassifyServer;
ros::Publisher pub1, pub_object, evidence_pub;
ros::ServiceClient get_object_position_client;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
bool match_found (false);
float model_ss_ (0.01f);
float scene_ss_ (0.03f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (3.0f);

/*
// purple   0   star, cross
// red      1   cube, hollow cube, sphere
// green    2   cube, cylinder
// blue     3   cube, triangle
// yellow   4   cube, sphere
// orange   5   star

//cube        0
//ball        1
//hollow cube 2
//cylinder    3
//triangle    4
//cross       5
//star        6
//battery     7
//unknown     8
*/

std::map<int, std::string> Color;
std::map<int, std::string> Shape;
std::map<std::string, std::string> ObjectID;

sensor_msgs::Image image;

int found_color, found_shape;

std::vector<std::vector<std::string> > model_lookup;
std::vector<std::string> purple;
std::vector<std::string> red;
std::vector<std::string> green;
std::vector<std::string> blue;
std::vector<std::string> yellow;
std::vector<std::string> orange;


pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);


std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
std::vector<pcl::Correspondences> clustered_corrs;

std::vector<float> x_points;
std::vector<float> y_points;
std::vector<float> z_points;


void camera_image_cb(const sensor_msgs::Image::ConstPtr& msg)
{
  image = *msg;
}


void Loader()
{
  purple.push_back("star.pcd");  purple.push_back("star2.pcd"); purple.push_back("cross.pcd"); purple.push_back("cross2.pcd");
  model_lookup.push_back(purple);
  red.push_back("cube.pcd"); red.push_back("cube2.pcd"); red.push_back("hollow_cube.pcd"); red.push_back("hollow_cube2.pcd"); 
  red.push_back("ball.pcd");
  model_lookup.push_back(red);
  green.push_back("cube.pcd");  green.push_back("cube2.pcd"); green.push_back("cylinder.pcd");
  model_lookup.push_back(green);
  blue.push_back("cube.pcd"); blue.push_back("cube2.pcd"); blue.push_back("triangle.pcd"); blue.push_back("triangle2.pcd");
  model_lookup.push_back(blue);
  yellow.push_back("cube.pcd"); yellow.push_back("cube2.pcd"); yellow.push_back("ball.pcd"); yellow.push_back("ball2.pcd");
  model_lookup.push_back(yellow);
  orange.push_back("star.pcd");
  model_lookup.push_back(orange);

  Color[0] = "purple";
  Color[1] = "red";
  Color[2] = "green";
  Color[3] = "blue";
  Color[4] = "yellow";
  Color[5] = "orange";

  Shape[0] = "cube";
  Shape[1] = "ball";
  Shape[2] = "hollow cube";
  Shape[3] = "cylinder";
  Shape[4] = "triangle";
  Shape[5] = "cross";
  Shape[6] = "star";
  Shape[7] = "battery";
  Shape[8] = "unknown";

  ObjectID["object"] = "An Object";
  ObjectID["red cube"] = "Red Cube";
  ObjectID["blue cube"] = "Blue Cube";
  ObjectID["green cube"] = "Green Cube";
  ObjectID["yellow cube"] = "Yellow Cube";
  ObjectID["yellow ball"] = "Yellow Ball";
  ObjectID["red ball"] = "Red Ball";
  ObjectID["green cylinder"] = "Green Cylinder";
  ObjectID["blue triangle"] = "Blue Triangle";
  ObjectID["purple cross"] = "Purple Cross";
  ObjectID["orange star"] = "Patric";
}


double computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  std::cout << "computeCloudResolution";
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}


double CalcMHWScore(std::vector<float> scores)
{
  double median;
  size_t size = scores.size();

  sort(scores.begin(), scores.end());

  if (size  % 2 == 0) {
    median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
  }
  else {
    median = scores[size / 2];
  }
  return median;
}


int find_shape(int color, int index)
{
  switch(color) {
    case 0:
      if (index == 0)
        return 6;
      else if (index == 1)
        return 6;
      else if (index == 2)
        return 5;
      else if (index == 3)
        return 5;
      else
        return 8;
    case 1:
      if (index == 0)
        return 0;
      else if (index == 1)
        return 0;
      else if (index == 2)
        return 2;
      else if (index == 3)
        return 2;
      else if (index == 4)
        return 1;
      else
        return 8;
    case 2:
      if (index == 0)
        return 0;
      else if (index == 1)
        return 0;
      else if (index == 2)
        return 3;
      else
        return 8;
    case 3:
      if (index == 0)
        return 0;
      else if (index == 1)
        return 0;
      else if (index == 2)
        return 4;
      else if (index == 3)
        return 4;
      else
        return 8;
    case 4:
      if (index == 0)
        return 0;
      else if (index == 1)
        return 0;
      else if (index == 2)
        return 1;
      else if (index == 3)
        return 1;
      else
        return 8;
    case 5:
      if (index == 0)
        return 6;
    default:
      return 8;
  }
}


void classify(const vision::ClassifyGoalConstPtr& goal, ClassifyServer* as) 
{
  ros::spinOnce();

  found_color = goal->color;
  found_shape = -1;

  custom_msgs::color_center srv;
  srv.request.i = found_color;

  sensor_msgs::PointCloud2 coloredCloud;

  if (get_object_position_client.call(srv)) {
    double x = (double)srv.response.x;
    double y = (double)srv.response.y;
    double z = (double)srv.response.z;
    coloredCloud = srv.response.coloredCloud;
  }

  /*
  std::stringstream ss;
  ss << "scene" << found_color << ".pcd";
  std::string str = ss.str();
  std::string path_scene = ros::package::getPath("vision") + "/pcd_files/scene/" + str;
  */

  pcl::fromROSMsg(coloredCloud, *scene);

  if (scene -> points.size() <= 10)  {
    as->setAborted();
    std::cerr << "hello akshay\n";
    return;
  }

  float max_percent = 0;

  for (int t=0; t<5; t++)
  {
    if (t >= model_lookup[found_color].size())
      break;

    std::string path_model = ros::package::getPath("vision") + "/pcd_files/" + model_lookup[found_color][t];

    if (pcl::io::loadPCDFile (path_model, *model) < 0)
    {
      std::cout << "Error loading model cloud." << std::endl;
      return;
    }
    else
    {
      std::cout << "Loaded " << model_lookup[found_color][t] << std::endl;
    }
    
    /*
    if (pcl::io::loadPCDFile (path_scene, *scene) < 0)
    {
      std::cout << "Error loading scene cloud." << std::endl;
      return;
    }
    else
    {
      std::cout << "Loaded scene" << found_color << std::endl;
    }
    */

    //  Set up resolution invariance
    if (use_cloud_resolution_)
    {
      float resolution = static_cast<float> (computeCloudResolution (model));
      if (resolution != 0.0f)
      {
        model_ss_   *= resolution;
        scene_ss_   *= resolution;
        rf_rad_     *= resolution;
        descr_rad_  *= resolution;
        cg_size_    *= resolution;
      }

      std::cout << "Model resolution:       " << resolution << std::endl;
      std::cout << "Model sampling size:    " << model_ss_ << std::endl;
      std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
      std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
      std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
      std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
    }

    //  Compute Normals
    std::cout << "Compute Normals" << std::endl;
    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setKSearch (10);

    norm_est.setInputCloud (model);
    norm_est.compute (*model_normals);

    norm_est.setInputCloud (scene);
    norm_est.compute (*scene_normals);
    
    //  Downsample Clouds to Extract keypoints
    std::cout << "Downsample Clouds to Extract keypoints";
    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud (model);
    uniform_sampling.setRadiusSearch (0.008);
    ROS_INFO("%f", model_ss_);
    //uniform_sampling.filter (*model_keypoints);
    pcl::PointCloud<int> keypointIndices1;
    uniform_sampling.compute(keypointIndices1);
    pcl::copyPointCloud(*model, keypointIndices1.points, *model_keypoints);
    std::cout << "Model total points: " << model->size () << std::endl;
    std::cout << "Selected Keypoints: " << model_keypoints->size() << std::endl;
    
    uniform_sampling.setInputCloud (scene);
    uniform_sampling.setRadiusSearch (0.008);
    ROS_INFO("%f", scene_ss_);
    //uniform_sampling.filter (*scene_keypoints);
    pcl::PointCloud<int> keypointIndices2;
    uniform_sampling.compute(keypointIndices2);
    pcl::copyPointCloud(*scene, keypointIndices2.points, *scene_keypoints);
    std::cout << "Scene total points: " << scene->size() << std::endl;
    std::cout << "Selected Keypoints: " << scene_keypoints->size() << std::endl;

    //  Compute Descriptor for keypoints
    pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    descr_est.setRadiusSearch (descr_rad_);

    descr_est.setInputCloud (model_keypoints);
    descr_est.setInputNormals (model_normals);
    descr_est.setSearchSurface (model);
    descr_est.compute (*model_descriptors);

    descr_est.setInputCloud (scene_keypoints);
    descr_est.setInputNormals (scene_normals);
    descr_est.setSearchSurface (scene);
    descr_est.compute (*scene_descriptors);

    //  Find Model-Scene Correspondences with KdTree
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    pcl::KdTreeFLANN<DescriptorType> match_search;
    match_search.setInputCloud (model_descriptors);

    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (size_t i = 0; i < scene_descriptors->size (); ++i)
    {
      std::vector<int> neigh_indices (1);
      std::vector<float> neigh_sqr_dists (1);
      if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
      {
        continue;
      }
      int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
      if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
      {
        pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
        model_scene_corrs->push_back (corr);  
      }
    }

    //  Using Hough3D
    if (use_hough_)
    {
      //  Compute (Keypoints) Reference Frames only for Hough
      pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
      pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

      pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
      rf_est.setFindHoles (true);
      rf_est.setRadiusSearch (rf_rad_);

      rf_est.setInputCloud (model_keypoints);
      rf_est.setInputNormals (model_normals);
      rf_est.setSearchSurface (model);
      rf_est.compute (*model_rf);

      rf_est.setInputCloud (scene_keypoints);
      rf_est.setInputNormals (scene_normals);
      rf_est.setSearchSurface (scene);
      rf_est.compute (*scene_rf);

      //  Clustering
      pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
      clusterer.setHoughBinSize (cg_size_);
      clusterer.setHoughThreshold (cg_thresh_);
      clusterer.setUseInterpolation (true);
      clusterer.setUseDistanceWeight (false);

      clusterer.setInputCloud (model_keypoints);
      clusterer.setInputRf (model_rf);
      clusterer.setSceneCloud (scene_keypoints);
      clusterer.setSceneRf (scene_rf);
      clusterer.setModelSceneCorrespondences (model_scene_corrs);

      //clusterer.cluster (clustered_corrs);
      clusterer.recognize (rototranslations, clustered_corrs);
      
      //compare.push_back(model_scene_corrs->size()/model_keypoints->size());

      std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;

      float percent = float(model_scene_corrs->size())/float(scene_keypoints->size());

      std::cout << "Percent: " << percent << std::endl;

      if (percent > 0.70) {
        match_found = true;
        found_shape = find_shape(found_color,t);
        break;
      }
      else if (percent > max_percent)  {
        found_shape = find_shape(found_color,t);
        max_percent = percent;
      }
    }
    else // Using GeometricConsistency
    {
      pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
      gc_clusterer.setGCSize (cg_size_);
      gc_clusterer.setGCThreshold (cg_thresh_);
      ROS_INFO("%f",cg_thresh_);

      gc_clusterer.setInputCloud (model_keypoints);
      gc_clusterer.setSceneCloud (scene_keypoints);
      gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

      //gc_clusterer.cluster (clustered_corrs);
      gc_clusterer.recognize (rototranslations, clustered_corrs);

      std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;

      float percent = float(model_scene_corrs->size())/float(scene_keypoints->size());

      std::cout << "Percent: " << percent << std::endl;

      if (percent > 0.70) {
        match_found = true;
        found_shape = find_shape(found_color,t);
        break;
      }
      else if (percent > max_percent)  {
        found_shape = find_shape(found_color,t);
        max_percent = percent;
      }
    }

  }


  if (max_percent > 0.7)  {
    match_found = true;
  }


  if (match_found)
  {
    //  Output results
    //
    std::cout << "Model instances found: " << rototranslations.size() << std::endl;
    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
      std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
      std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

      // Print the rotation matrix and translation vector
      Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
      Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);
    }

    std::cerr << "Color : " << found_color << " Shape : " << found_shape << std::endl;

    std::stringstream color_and_shape;
    color_and_shape << Color[found_color] << " " << Shape[found_shape];
    std::string object_id = ObjectID[color_and_shape.str()];

    std_msgs::String text;
    std::stringstream ss;
    ss << "I see a " << color_and_shape.str();
    text.data = ss.str();
    pub1.publish(text);


    ras_msgs::RAS_Evidence evidence_msg;

    evidence_msg.stamp = ros::Time::now();
    evidence_msg.group_number = 5;
    evidence_msg.image_evidence = image;
    evidence_msg.object_id = object_id;

    evidence_pub.publish(evidence_msg);

    for (size_t k = 0; k < rototranslations.size (); ++k)
    {
      for (size_t j=0; j<clustered_corrs[k].size (); j++)
      {      
        float x = scene_keypoints->at(clustered_corrs[k][j].index_match).x;
        float y = scene_keypoints->at(clustered_corrs[k][j].index_match).y;
        float z = scene_keypoints->at(clustered_corrs[k][j].index_match).z;

        x_points.push_back(x);
        y_points.push_back(y);
        z_points.push_back(z);
      }
    }

     
    double x_sum = std::accumulate(x_points.begin(), x_points.end(), 0.0);
    double x_mean = x_sum / x_points.size();

    double y_sum = std::accumulate(y_points.begin(), y_points.end(), 0.0);
    double y_mean = y_sum / y_points.size();

    double z_sum = std::accumulate(z_points.begin(), z_points.end(), 0.0);
    double z_mean = z_sum / z_points.size();


    if (!isnan(x_mean))
    {
      ROS_INFO("x: %f", x_mean);
      ROS_INFO("y: %f", y_mean);
      ROS_INFO("z: %f", z_mean);
    }

    tf::TransformListener listener;
    tf::StampedTransform transform;

    geometry_msgs::PointStamped obj_camera;
    obj_camera.header.stamp = ros::Time();
    obj_camera.header.frame_id = "/camera_link";
    obj_camera.point.x = x_mean;
    obj_camera.point.y = y_mean;
    obj_camera.point.z = z_mean;

    geometry_msgs::PointStamped obj_base;
    try
    {
      listener.waitForTransform("/base_link", "/camera_link", obj_camera.header.stamp, ros::Duration(1.0));
      listener.transformPoint("/base_link", obj_camera, obj_base);
    }
    catch(tf::TransformException& ex)
    {
      ROS_ERROR("Received an exception trying to transform a point from \"camera_link\" to \"base_link\": %s", ex.what());
    }

    if (!isnan(x_mean))
    {
      custom_msgs::Object object;
      object.detectionSource = "SHAPE_CANDY";
      object.position.x = obj_base.point.x;
      object.position.y = obj_base.point.y;
      object.position.z = obj_base.point.z;
      object.color = found_color;
      object.shape = found_shape;
      object.weight = 1;

      //pub_object.publish(object);

      ROS_INFO("x: %f", obj_base.point.x);
      ROS_INFO("y: %f", obj_base.point.y);
      ROS_INFO("z: %f", obj_base.point.z);

      as->setSucceeded();
    }
    else  {
      as->setAborted();
    }

    x_points.clear();
    y_points.clear();
    z_points.clear();

    return;
  }
  else
  {
    std_msgs::String text;
    std::stringstream ss;
    ss << "I see a " << Color[found_color] << " object";
    text.data = ss.str();
    pub1.publish(text);

    ras_msgs::RAS_Evidence evidence_msg;

    evidence_msg.stamp = ros::Time::now();
    evidence_msg.group_number = 5;
    evidence_msg.image_evidence = image;
    evidence_msg.object_id = ObjectID["object"];

    evidence_pub.publish(evidence_msg);
  }

  found_color = -1;
  found_shape = -1;

  as->setAborted();
}



int main (int argc, char *argv[])
{
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;

  ros::Subscriber image_sub = nh.subscribe ("/camera/rgb/image_color", 1, camera_image_cb);

  // Create a ROS publisher for object point
  pub1 = nh.advertise<std_msgs::String> ("espeak/string", 1);
  //pub_object = nh.advertise<custom_msgs::Object> ("detected_objects", 1);
  evidence_pub = nh.advertise<ras_msgs::RAS_Evidence> ("evidence", 1);

  get_object_position_client = nh.serviceClient<custom_msgs::color_center>("get_color_center");

  Loader();

  ClassifyServer classifyServer(nh, "classify", boost::bind(&classify, _1, &classifyServer), false);
  classifyServer.start();

  // Spin
  ros::spin ();
}
