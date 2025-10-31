#include "perception/perception_node.hpp"

#include <functional>
#include <string>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"

namespace perception
{
PerceptionNode::PerceptionNode()
: rclcpp::Node("perception_node")
{
  declare_parameter<std::string>("image_topic", "/camera/camera/color/image_raw/compressed");
  declare_parameter<std::string>("window_name", "PerceptionView");

  const auto image_topic = get_parameter("image_topic").as_string();
  window_name_ = get_parameter("window_name").as_string();

  cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);

  image_subscription_ = create_subscription<sensor_msgs::msg::CompressedImage>(
    image_topic, rclcpp::SensorDataQoS(),
    std::bind(&PerceptionNode::on_image, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "Perception node subscribing to %s", image_topic.c_str());
}

PerceptionNode::~PerceptionNode()
{
  if (!window_name_.empty())
  {
    cv::destroyWindow(window_name_);
  }
}

void PerceptionNode::on_image(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
{
  if (msg->data.empty())
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Received compressed image with empty data buffer");
    return;
  }

  try
  {
    cv::Mat decoded = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (decoded.empty())
    {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000, "OpenCV failed to decode compressed image");
      return;
    }

    cv::imshow(window_name_, decoded);
    cv::waitKey(1);  // allow OpenCV to process window events
  }
  catch (const cv::Exception & e)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000, "OpenCV exception during decode: %s", e.what());
  }
}
}  // namespace perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<perception::PerceptionNode>());
  rclcpp::shutdown();
  return 0;
}
