[[BevFormer]]

##### Intro
​​2020年​​《Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D》论文在计算机视觉顶会（如ECCV/CVPR）发布，首次提出Lift-Splat-Shoot（LSS）框架，解决了多相机图像到鸟瞰图（BEV）的转换问题。

##### 流程简述
backbone特征提取、多尺度特征图融合、3d

##### 关键结论


目标检测：
CenterPoint：中心点优先，属性绑定，1x1卷积预测，高斯化真值中心点坐标，NMS去重


车道线检测：
1. 栅格属性分割
2. 关键点预测？不是太理解
3. DETR范式，每个实例一个query，解码得到控制点坐标（多段线或者贝塞尔曲线）
4. 自回归预测？具体如何操作？
