#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
using namespace cv;
using namespace cv::dnn;
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;
/* Find best class for the blob (i. e. class with maximal probability) */
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
int main(int argc, char **argv)
{
CV_TRACE_FUNCTION();
String modelTxt = "test.prototxt";
String modelBin = "finetune_squeezenet__iter_96000.caffemodel";
String imageFile = "fcwDayLight021418.jpg";
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
Mat testimg = imread(imageFile);
if (testimg.empty())
{
std::cerr << "Can't read image from the file: " << imageFile << std::endl;
exit(-1);
}
Mat img = testimg(Rect(cvPoint(420,180),cvSize(117,115)));
//GoogLeNet accepts only 224x224 BGR-images
double t1 = (double)getTickCount();
Mat inputBlob = blobFromImage(img, 1.0f, Size(227, 227),Scalar(104, 117, 123), false); //Convert Mat to batch of images
CV_TRACE_REGION("forward");
net.setInput(inputBlob, "data"); //set the network input
t1 = ((double)getTickCount()-t1)/getTickFrequency()*1000;
cout<<"read an image to input blob time is "<< t1 << "ms"<<endl;
Mat prob;
cv::TickMeter t;
for (int i = 0; i < 10; i++)
{
CV_TRACE_REGION("forward");
net.setInput(inputBlob, "data"); //set the network input
t.start();
prob = net.forward("prob"); //compute output
t.stop();
}
int classId;
double classProb;
getMaxClass(prob, &classId, &classProb);//find the best class
std::vector<String> classNames = readClassNames();
std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;
char preProb[100];
sprintf(preProb,"%f%s",classProb*100,"%");
String tx = classNames.at(classId)+String(preProb);
if(classId==0)
{
	putText(testimg,tx,Point(50,50),2,1,Scalar(0,0,255),1);
}
else if(classId==1)
{
	putText(testimg,tx,Point(50,50),2,1,Scalar(0,255,0),1);
}
else
{
	putText(testimg,tx,Point(50,50),2,1,Scalar(255,0,0),1);
}
imshow("preResult",testimg);
cvWaitKey(10000);
return 0;
} //main

