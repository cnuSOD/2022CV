import datetime
import logging
import pathlib
from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from .common import Face, FacePartsName, Visualizer#导入文件中的函数
from .gaze_estimator import GazeEstimator
from .utils import get_3d_face_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}#实时检测时按q退出

    def __init__(self, config: DictConfig):#初始化函数
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

        self.cap = self._create_capture()#cap是实时监测或者视频文件的截图
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

    def run(self) -> None:#检测视频或者图片文件路径
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()#如果视频文件路径存在或实时摄像头检测，则对视频进行视线估计
        elif self.config.demo.image_path:
            self._run_on_image()#如果图片文件路径存在，则对图片进行视线估计
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)#读取图像
        self._process_image(image)
        if self.config.demo.display_on_screen:#配置演示屏幕显示
            while True:
                key_pressed = self._wait_key()#定义交互按键
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)#显示图片
        if self.config.demo.output_dir:#配置输出
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name#设置输出路径
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)#读取路径及图片重命名

    def _run_on_video(self) -> None:
        while True:
            if self.config.demo.display_on_screen:#配置演示屏幕显示
                self._wait_key()
                if self.stop:#如果按下q退出键，则退出
                    break

            ok, frame = self.cap.read()#ok表示是否读取到图片，frame表示截取到一帧的图像
            if not ok:#若没读取到，结束
                break
            self._process_image(frame)

            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)#将图像矩阵显示在一个窗口中
        self.cap.release()#释放资源并关闭窗口
        if self.writer:
            self.writer.release()

    def _process_image(self, image) -> None:#处理每一帧的图像
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)#检测人脸
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)#绘画脸部包围框
            self._draw_head_pose(face)#绘制模型坐标系的坐标轴
            self._draw_landmarks(face)#绘制地标
            self._draw_face_template_model(face)#绘制面模板模型
            self._draw_gaze_vector(face)#绘制凝视向量
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:#逐帧读取视频
        if self.config.demo.image_path:#如果对图像处理，则不需要截屏，返回空
            return None
        if self.config.demo.use_camera:#如果是实时监测，进行截屏
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:#如果存在视频路径，则对路径中的视频进行帧获取
            cap = cv2.VideoCapture(self.config.demo.video_path)#逐帧读取路径中的视频文件
        else:
            raise ValueError #若都不符合，则提示错误
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:#创造导出文件夹
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:#记录时间并且返回
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:#如果有图片读取的路径则返回空
            return None
        if not self.output_dir:#如果没有输出文件路径则返回空
            return None
        ext = self.config.demo.output_file_extension#定义ext为配置示范文件的拓展
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')#如果是mp4文件，配置为H264编码类型
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')#如果是avi文件，配置为MPEG-1编码类型
        else:
            raise ValueError
        if self.config.demo.use_camera:#如果使用外摄，导出视频，命名格式为时间戳加拓展名
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:#如果用路径中的视频文件，同上
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name#读取文件输出路径
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))#输出文件为视线估计照片的长宽为基础的四字编码后的文件
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:#快捷键
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox#按下b显示或者去除脸部包围框
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks#按下l显示或者去除地标
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose#按下h显示或者去除面部坐标系的坐标轴
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image#按下n打开或者暂停面部图像
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model#显示模板模型
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:#绘制头部姿势选择框
        if not self.show_head_pose:
            return
        length = self.config.demo.head_pose_axis_length#绘制模型坐标系的轴
        self.visualizer.draw_model_axes(face, length, lw=2)#绘制选择框的标准

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == 'MPIIGaze':#如果选择MPIIGAZE模型
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image#分别得到左右眼的图像
            normalized = np.hstack([reye, leye])
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:#如果选择MPIIFaceGaze或ETH-XGaze模型
            normalized = face.normalized_image#脸部归一化图像
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)#展示归一化图像

    def _draw_gaze_vector(self, face: Face) -> None:#画出视线估计方向
        length = self.config.demo.gaze_visualization_length#所画视线长度
        if self.config.mode == 'MPIIGaze':
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)#可视化工具
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))#计算水平和竖直方向的角度
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError
