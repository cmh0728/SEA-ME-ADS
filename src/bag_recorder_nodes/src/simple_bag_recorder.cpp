#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>

#include <sensor_msgs/msg/compressed_image.hpp>

#include <rosbag2_cpp/writer.hpp>

using std::placeholders::_1;

class SimpleBagRecorder : public rclcpp::Node
{
public:
  SimpleBagRecorder()
  : Node("simple_bag_recorder")
  {
    // rosbag2 writer 생성
    writer_ = std::make_unique<rosbag2_cpp::Writer>();

    // bag 파일 폴더 이름 (공백 없는 이름을 권장)
    // Humble에서는 이 open(const std::string&)은 아직 써도 됨.
    writer_->open("final_test1");

    // /camera/camera/color/image_raw/compressed 토픽 구독
    subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "/camera/camera/color/image_raw/compressed",
      10, // 필요하면 rclcpp::SensorDataQoS() 로 바꿀 수도 있음
      std::bind(&SimpleBagRecorder::topic_callback, this, _1));
  }

private:
  void topic_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    // 1) sensor_msgs::msg::CompressedImage → rclcpp::SerializedMessage 직렬화
    rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serializer;

    // ★ 새 API에 맞게 shared_ptr<SerializedMessage> 사용
    auto serialized_msg = std::make_shared<rclcpp::SerializedMessage>();
    serializer.serialize_message(msg.get(), serialized_msg.get());

    // 2) 타임스탬프 (수신 시각 사용)
    rclcpp::Time time_stamp = this->now();

    // 3) bag에 쓰기 - 새 write API 사용
    //
    // void write(std::shared_ptr<rclcpp::SerializedMessage> message,
    //            const std::string & topic_name,
    //            const std::string & type_name,
    //            const rclcpp::Time & time);
    writer_->write(
      serialized_msg,
      "/camera/camera/color/image_raw/compressed",
      "sensor_msgs/msg/CompressedImage",
      time_stamp);
  }

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
  std::unique_ptr<rosbag2_cpp::Writer> writer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimpleBagRecorder>());
  rclcpp::shutdown();
  return 0;
}
