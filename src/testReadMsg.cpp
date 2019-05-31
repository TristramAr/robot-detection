#include <ros/ros.h>
#include<std_msgs/Int32.h>
#include<std_msgs/String.h>
#include<geometry_msgs/Point.h>
#include <robot_tensorflow/detection.h>
#include <robot_tensorflow/objInfo.h>
#include<vector>
#include<iostream>
using namespace std;

vector<robot_tensorflow::detection> objInfos; 
void testReadMsg(const robot_tensorflow::objInfo &msg )
{
    ROS_INFO("i received the msg");
    int num = msg.objInfos.size();
    objInfos.clear();
    objInfos.resize(num);
    for(int i=0; i<num; i++)
    {
        // int num = objectData.objInfos.size();

        // vector <robot_tensorflow::detection> objInfos; 
        objInfos[i] = msg.objInfos[i];
        ROS_INFO("i get the result %d",msg.objInfos[i].px);
    }
    
}
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "readMsg");
    ros::NodeHandle nh;
    //ros::Subscriber sub = n.Subscriber("objectData",1000,testReadMsg);
    //ros::Subscriber sub = nh.subscribe("topic_name", 1000, subCallback);
    ros::Subscriber sub = nh.subscribe("result", 1, testReadMsg);
    
    ros::spin();
    return 0;
}