1.通过Anaconda安装pytorch

2.把ptgaze包安装下来（pip install ptgaze 已经设置了setup文件）

3.输入conda activate pytorch进入创建的虚拟环境pytorch,当发现命令行最左端有
(pytorch)，则说明已经成功进入ptorch

4.使用
1).基本运行指令
ptgaze -mode eth-xgaze
该指令默认使用的是eth-gaze数据集训练的模型，人脸识别模型为mediapipe，CNN网络为resnet18，检测方式为实时检测。
ptgaze -mode mpiigaze
该指令默认使用的是mpiigaze数据集训练的模型，人脸识别为dlib模型，CNN网络为resnet_preact，检测方式为实时检测。
ptgaze -mode mpiifacegaze
该指令默认使用的是mpiifacegaze数据集训练的模型，人脸识别为dlib模型，CNN网络为resnet_simple，检测方式为实时检测。
2).高级运行指令
通过以下表格可选择自己想要的效果
[--mode {mpiigaze, mpiifacegaze, eth-xgaze}]
[--face-detector{dlib,face_alignment_dlib,face_alignment_sfd,
mediapipe}]
[--image IMAGE] [--video VIDEO] [--camera CAMERA]
说明：若想使用mpiigaze数据集训练的模型，并且使用mediapipe的人脸识别模型，可在命令行输入指令ptgaze -mode mpiigaze -face-detector mediapipe即可。(进行图片或者视频检测的话需要在yaml源文件中预设图片或者视频的路径，并且要添加输出的路径，否则就会报错)
在处理图像或视频时，按下窗口上的下列键显示或隐藏中间结果:
- ‘l’:地标
- ‘h’:头部姿势
- ‘t’:3D人脸模型的投影点
- ‘b’:脸包围框
- ‘n’:人脸检测
