#include "opencv2/core/core.hpp"  
#include <opencv2/dnn.hpp>
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <opencv2/core/utils/trace.hpp>
#include <fstream>  
#include <iostream>  
#include <stdio.h>  
#include <io.h>

using namespace std;  
using namespace cv;  
using namespace cv::dnn;

void on_Trackbar( int, void* );
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb);
static std::vector<String> readClassNames(const char*);

string windowName = "detect_and_show";
VideoCapture capture;
int currentFrame = 0;

int main()
{
	Mat frame,ROI;
	CV_TRACE_FUNCTION();
	String modelTxt = "test.prototxt";
	String modelBin = "finetune_squeezenet__iter_96000.caffemodel";
	Net net;
	try {
	net = dnn::readNetFromCaffe(modelTxt, modelBin);
	}
	catch (cv::Exception& e) {
	std::cerr << "Exception: " << e.what() << std::endl;
	if (net.empty())
	{
	std::cerr << "Can't load network by using the following files: " << std::endl;
	std::cerr << "prototxt: " << modelTxt << std::endl;
	std::cerr << "caffemodel: " << modelBin << std::endl;
	exit(-1);
	}
	}
	ifstream videoName("/home/nvidia/opencv_test/traffic_light_test/test_videos.txt");
	int roi_x      = 20;
	int roi_y      = 200;
	int roi_width  = 650;
	int roi_height = 250;
	namedWindow(windowName);
	while(!videoName.eof())		      //文件打开成功则执行
	{
		char ch[200];
		videoName.getline(ch,200);    //从文件中获取字符串存入字符变量ch中
		cout << ch << endl;
		capture.open(ch);
		if (capture.isOpened())
		{
			int frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);      //获取当前视频总帧数
			//在创建的窗体中创建一个滑动条控件
			char TrackbarName[50];
			sprintf( TrackbarName, "当前帧" );
			createTrackbar( TrackbarName, windowName, &currentFrame, frameCount,on_Trackbar);
			for (;;)
			{
				capture >> frame;
				if (frame.empty())
				{
					break;
				}
				//结果在回调函数中显示
				on_Trackbar( currentFrame, 0 );
				setTrackbarPos(TrackbarName,windowName,currentFrame);
				capture.set(CV_CAP_PROP_POS_FRAMES,currentFrame++);
				if (currentFrame >= frameCount)
				{
					break;
				}
				int key=cvWaitKey(10);//读取键值到key变量中
				//根据key变量的值，进行不同的操作
				switch(key)
				{
					//======================【程序退出相关键值处理】=======================  
				case 27://按键ESC
					return 0;
					break; 

					//======================【roi移动相关键值处理】=======================  
				case 'u':   //按键U按下，roi左上点上移，宽高不变
					{
						roi_y = roi_y - 5;
						if (roi_y < 0)
						{
							roi_y = 0;
							cout<<"roi_y == 0，roi不能再向上移动了！！！"<<endl;
						}
					}
					break; 

				case 'd':   //按键d按下，roi左上点下移
					{
						roi_y = roi_y + 5;
						if ((roi_y + roi_height)>frame.rows)
						{
							roi_height = frame.rows - roi_y;
							cout <<"roi_y + roi_height大于图片高度，roi不能向下移动了！！！"<<endl;
						}
					}
					break; 

				case 'l':   //按键l按下，roi左上点左移
					{
						roi_x = roi_x - 5;
						if (roi_x < 0)
						{
							roi_x = 0;
							cout<<"roi_x == 0，roi不能再向左移动了！！！"<<endl;
						}
					}
					break; 

				case 'r':  //按键r按下，roi左上点右移
					{
						roi_x = roi_x + 5;
						if ((roi_x + roi_width)>frame.cols)
						{
							roi_width = frame.cols - roi_x;
							cout <<"roi_x + roi_width大于图片宽度，roi不能向右移动了！！！"<<endl;
						}
					}
					break; 

				case 'w':   //按W键，增加roi宽度
					{
						roi_width = roi_width + 5;
						if ((roi_x + roi_width)>frame.cols)
						{
							roi_width = frame.cols - roi_x;
							cout <<"roi_x + roi_width大于图片宽度，不能再增加roi_width了！！！"<<endl;
						}
					}
					break; 

				case'h':   //按h键，增加roi高度
					{
						roi_height = roi_height + 5;
						if ((roi_y + roi_height)>frame.rows)
						{
							roi_height = frame.rows - roi_y;
							cout <<"roi_y + roi_height大于图片高度，不能再增加roi_height了！！！"<<endl;
						}
					}
					break; 

				case'z':   //按z键，减少roi宽度
					{
						roi_width = roi_width - 5;
					}
					break; 

				case'x':   //按x键，减少roi高度
					{
						roi_height = roi_height - 5;
					}
					break;

				case 'p':
					{
					   getchar();
					}
					break;

				default:
					break;
				}
				
				rectangle(frame,Rect(roi_x,roi_y,roi_width,roi_height),Scalar(255,255,0),1);
				
				ROI = frame(Rect(roi_x,roi_y,roi_width,roi_height));
				double t1 = (double)getTickCount();
				Mat inputBlob = blobFromImage(ROI, 1.0f, Size(227, 227),Scalar(104, 117, 123), false);    //Convert Mat to batch of images
				CV_TRACE_REGION("forward");
				net.setInput(inputBlob, "data"); //set the network input
				t1 = ((double)getTickCount()-t1)/getTickFrequency()*1000;
				cout<<"read an image to input blob time is "<< t1 << "ms"<<endl;
				Mat prob;
				//cv::TickMeter t;
				//for (int i = 0; i < 10; i++)
				//{
				//CV_TRACE_REGION("forward");
				//net.setInput(inputBlob, "data"); //set the network input
				//t.start();
				prob = net.forward("prob"); //compute output
				//t.stop();
				//}
				int classId;
				double classProb;
				getMaxClass(prob, &classId, &classProb);    //find the best class
				std::vector<String> classNames = readClassNames();
				std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
				std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
				//std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;
				char preProb[100];
				sprintf(preProb,"%f%s",classProb*100,"%");
				String tx = classNames.at(classId)+String(preProb);
				if(classId==0)
				{
					putText(frame,tx,Point(50,50),2,1,Scalar(0,0,255),1);
				}
				else if(classId==1)
				{
					putText(frame,tx,Point(50,50),2,1,Scalar(0,255,0),1);
				}
				else
				{
					putText(frame,tx,Point(50,50),2,1,Scalar(255,0,0),1);
				}
				imshow(windowName,frame);
				cvWaitKey(1);
			}
			currentFrame = 0;
		}	
		capture.release();
	}
	videoName.close();
	return 0;
}

void on_Trackbar( int pos, void* )
{
	pos = capture.get(CV_CAP_PROP_POS_FRAMES);
}

static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
Point classNumber;
minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
*classId = classNumber.x;
}

static std::vector<String> readClassNames(const char *filename = "label.txt")
{
std::vector<String> classNames;
std::ifstream fp(filename);
if (!fp.is_open())
{
std::cerr << "File with classes labels not found: " << filename << std::endl;
exit(-1);
}
std::string name;
while (!fp.eof())
{
std::getline(fp, name);
if (name.length())
classNames.push_back( name.substr(name.find(' ')+1) );
}
fp.close();
return classNames;
}