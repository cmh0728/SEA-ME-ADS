#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <std_msgs/msg/string.hpp>

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

    // 그냥 이름만 주고 open (기본 storage 옵션: sqlite3)
    writer_->open("my_bag");

    // String 타입으로 평범하게 subscribe
    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "chatter", 10,
      std::bind(&SimpleBagRecorder::topic_callback, this, _1));
  }

private:
  void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
  {
    // 1) std_msgs::msg::String → rclcpp::SerializedMessage 로 직렬화
    rclcpp::Serialization<std_msgs::msg::String> serializer;
    rclcpp::SerializedMessage serialized_msg;
    serializer.serialize_message(msg.get(), &serialized_msg);

    // 2) 타임스탬프 (수신 시각 사용)
    rclcpp::Time time_stamp = this->now();

    // 3) bag에 쓰기
    //    Writer::write(const rclcpp::SerializedMessage&, const std::string& topic,
    //                  const std::string& type, const rclcpp::Time& time)
    writer_->write(
      serialized_msg,
      "chatter",
      "std_msgs/msg/String",
      time_stamp);
  }

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  std::unique_ptr<rosbag2_cpp::Writer> writer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimpleBagRecorder>());
  rclcpp::shutdown();
  return 0;
}
