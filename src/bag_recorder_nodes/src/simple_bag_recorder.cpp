#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>

#include <sensor_msgs/msg/compressed_image.hpp>   // ★ 추가

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
    writer_->open("final_test1");

    // /camera/camera/color/image_raw/compressed 토픽 구독
    subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "/camera/camera/color/image_raw/compressed",
      10, // 필요하면 rclcpp::SensorDataQoS() 로 바꿀 수도 있음
      std::bind(&SimpleBagRecorder::topic_callback, this, _1));
  }

private:
  // ★ const 제거 (writer_ 사용 때문에)
  void topic_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    // 1) sensor_msgs::msg::CompressedImage → rclcpp::SerializedMessage 직렬화
    rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serializer;
    rclcpp::SerializedMessage serialized_msg;
    serializer.serialize_message(msg.get(), &serialized_msg);

    // 2) 타임스탬프 (수신 시각 사용)
    rclcpp::Time time_stamp = this->now();

    // 3) bag에 쓰기
    //    Writer::write(const rclcpp::SerializedMessage&, const std::string& topic,
    //                  const std::string& type, const rclcpp::Time& time)
    writer_->write(
      serialized_msg,
      "/camera/camera/color/image_raw/compressed",   // 토픽 이름
      "sensor_msgs/msg/CompressedImage",            // 타입 이름 (중요)
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
