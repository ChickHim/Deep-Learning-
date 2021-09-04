import os, sys, time

"""
    [ 概念 ]
        1. Seq2seq 模型
            -> 结构  Encoder-Decoder
                输入给 Encoder 的 RNN, 最后输出一个隐含状态 作为 Decoder 的 RNN 初始状态
                此时输入给 Decoder 一个 Null, Decoder 会生成第一个单词, 第一个单词 输入给 Decoder 会生成第二个单词
                ...
                
                只要是 RNN 就可以, 例如 LTSM, GRU, CIFG....
                
                    GRU 是 LSTM 的一个变种
                    LSTM 遗忘门 输入门 输出门
                    GRU 把 遗忘门 输入门 变成一个, 它认为 遗忘门 + 输入门 = 1
                
            -> 缺点
                定长编码是信息瓶颈
                长度越长，前面输入进 RNN 的信息就会被稀释
        
        2. Attention + Seq2seq
            -> 结构  Encoder-Decoder
                Encoder 每一步的输出都会参与到 Decoder 每一步生成的计算中
            
            -> Attention
            
                可以看作是以一种无损的信息传递方式
                
                1-1. Bahdanau 注意力  (常用)
                score = FC(tanh(FC(EO) + FC(H)))
                    score : 长度和 EO 相同的一维向量
                    EO : Encoder各个位置的输出
                    H  : Decoder某一步的隐含状态
                    FC : 全连接层
                    X  : Decoder的一个输入
                    
                1-2. luong 注意力
                score = EO * W * H
                
                2. attention_weights = softmax(score, axis = 1)
                3. context = sum(attention_weights * EO, axis = 1)
                4. final_input = concat(context, embed(x))
                
            -> 改良
                去除定长的编码瓶颈, 信息无损从 Encoder 到 Decoder
                每一步都会使用所有信息的权重，减轻稀释程度
            
            -> 缺点
                采用 RNN 变种, 计算依然有瓶颈, 并行度不高
                只有 Encoder 和 Decoder 之间有 attention
                
            
            
        3. Transformer 模型 (Bert, GPT, XLNet 都是基于 Transformer)
        -> 结构  多层 Encoder-Decoder
                 位置编码        -> 因为不适用 RNN, 使用并行处理机制,  
                 多头注意力      -> 缩放点积注意力
                                        公式 -> Attention(Q, K, V) = softmax(Q * K.T / (Dk ** 0.5) ) * V
                                        为什么除以 Dk -> 防止内积总和过大
                                        
                 Add & norm     -> Add 相当于 残差连接,  norm 相当于 归一化过程
                
            
        
"""
