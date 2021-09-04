import os, sys, time

"""
    [ 概念 ]
        神经网络的容量: 神经网络可以学习到的信息量
            （数据同样拥有信息量）
            网络容量 > 数据信息量 -> 网络可以完全记忆样本数据 -> 过拟合 缺乏泛化能力
            网络容量 < 数据信息量 -> 欠拟合 准确率低 
            两者差不多 -> 效果最好
    
    [ 内容 ]
    一、CNN: 卷积神经网络
        -> 通过 局部连接、参数共享 解决 参数过多的问题
            局部连接有效性
                依靠 图像本身的区域性
                例如 嘴部附近的值是比较接近的， 脸颊附近的值也是比较接近的
            
            参数共享
                不同的局部连接的参数是一样的
                
                依靠 图像特征与位置是无关的
                例如 一张脸放在图像的哪个位置都是脸 同时又解决了一部分过拟合
            
            参数过多导致
                1. 计算资源不足
                2. 容易过拟合 需要更多的训练数据
        
        结构: 
            ( 卷积层 + (可选) 池化层 ) * N + 全连接层 * M 
                -> N >= 1, M >= 0
                全连接前需要将数据展平 并且只能在 卷积池化之后
                    ->  卷积池化需要多维数据  全连接使用一维向量
                        多维数据展平至一维向量后 失去维度信息  全连接层无法重建维度信息
            
                全连接层可输出值或者向量
                    ->  回归预测或分类
                        但CNN多用于分类
            
            
    二、全卷积神经网络
        -> 全连接层换为反卷积层 解决 数据 size 逐渐变小
        
        结构: 
            ( 卷积层 + (可选) 池化层 ) * N + 反卷积层 * K 
                -> N >= 1, M >= 0
                反卷积层是卷积层的逆向操作 但也是卷积
                
                卷积池化会使得数据 size 越来越小 反卷积操作可使得数据 size 回到原来大小
                    ->  凭此原理可用于 物体分割
                
    三、深度可分离卷积
        以极小的精度损失 大幅度减小 计算量和内存的占用 使得CNN可以再手机等小型设备上运行
    
    四、
        数据增强: 在数据方面做一些操作使得神经网络训练效果提升
        迁移学习: 用之前训练的模型在当前任务上提升
        
    五、CNN 领域经典论文
    
        AlexNet 
                -> 深度学习的引爆点 第一个基于深度学习做图像分类的网络 并有较好的效果
        VGG Net 
                -> 比 AlexNet 更深的网络结构 提出经过一个 pooling 层 卷积通道翻倍 以及图像增强的手段
        NIN     
                -> (network in network) 提出 Global pooling 的概念 避免像 AlexNet VGG 使用全连接层
        ResNet  
                -> cnn 的变异 但被引用借鉴到不少其他领域 比如 强化学习(RL) 循环神经网络(RNN)
        Inception V1 ~ V4
                -> 都是由 Google 开发  V1 版本就是 著名的 Google Net
                   Google 使用 工程性思维 把卷积神经网络做到极致
        MobileNet
                -> 提出削减办法 减低硬件要求 比如 可分离神经网络
        NASNet
                -> 使用 AutoML 的技术设计了一种网络结构 达到了更高的水准
        ShakeShake
                -> 更像是一种正则化方法 对数据过拟合做出了一些处理
        
"""