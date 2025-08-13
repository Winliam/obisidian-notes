[[Transformer]]
[[DETR]]
[[DCN]]

##### Intro
针对DETR的两个显著局限性：
1. 小目标检测差（因为没有设计类似FPN的多尺度）
2. 收敛慢（因为随机初始化的Queries缺乏位置先验）
2021年ICLR上提出 **《Deformable DETR: Deformable Transformers for End-to-End Object Detection》**

##### 关键结论
1. 保留Resnet backbone输出的多层特征图，而不是DETR中的一层，并用256通道的1x1卷积调整到相同维度
2. 在进行多头自注意力编码之前，为特征图叠加三角函数形式的二维位置编码和可学习的尺度嵌入向量
3. 计算损失时，将一张图片中的真值框信息，用void填充到100个，在数量上与模型输出对齐。然后使用匈牙利匹配进行真值框与检出框的配对，然后针对匹配成功的pair即可计算分类损失和交点坐标回归损失。
4. 局限性：
	1. 小目标检测差（因为没有设计类似FPN的多尺度）
	2. 收敛慢（因为随机初始化的Queries缺乏位置先验）

