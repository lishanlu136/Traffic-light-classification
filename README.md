# Traffic-light-classification
## 说明
- 该项目用于对交通信号灯进行分类（3类：红灯，绿灯，背景）
- 用于训练的数据库为[vivachallenge-dataset](http://cvrr.ucsd.edu/vivachallenge/index.php/traffic-light/traffic-light-detection/)和自己录的视频数据库的组合
- 准备训练标签的程序在prepare_data文件夹里面
- 训练网络在Networks里面
- traffic_test_use_opencv文件夹是利用opencv-contrib里面的dnn模块读取caffemodel用于测试
- train_net.bat用于训练网络， classification.bat用于对单张图片测试
- 工程参考：[deep-learning-traffic-lights](https://github.com/davidbrai/deep-learning-traffic-lights)
- 知乎专栏文章参考：https://zhuanlan.zhihu.com/p/24955921
