# 答案抽取
### Question Answering System -- Answer extraction

## 说明
>本项目实现了一个单一事实类问答系统.何谓"单一事实类"?即,<u>回答的答案只为单一实体.</u>这里提供了两个 *深度学习模型* 来完成这一目标.(1)`BiLSTM-CRF`(它是由 [IDL](https://arxiv.org/abs/1607.06275) 提出的)(2)`Match-LSTM & CRF`(在前者的基础上添加了 `Match-LSTM` ,用于学习文本推理信息).项目中这两个模型都是使用 [PaddlePaddle](http://ai.baidu.com/paddlepaddle) 实现的.

## 安装
1. 搭建 `PaddlePaddle` 环境
    - 安装 `Docker` , 详见 [Docker docs](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
    - 获取 `PaddlePaddle` 的 Docker 镜像
    > `sudo docker pull docker.paddlepaddlehub.com/paddle`  
    > ps: 详见 [PaddlePaddle 使用文档](http://staging.paddlepaddle.org/docs/develop/documentation/fluid/zh/build_and_install/docker_install_cn.html) , 若有问题可使用邮箱联系我,我有制作好的 Docker 镜像;  
    > ps: 推荐使用 [`pycharm 远程调试 PaddlePaddle 代码`](https://blog.csdn.net/qq_26819733/article/details/75676098)
2. 下载运行所需资源
    - 词向量相关( data/embedding 文件夹下)
        - [`wordvecs.txt`](https://inotwant-picture-public.oss-cn-beijing.aliyuncs.com/%E6%AF%95%E8%AE%BE%E7%9B%B8%E5%85%B3/%E6%9D%90%E6%96%99/embedding/wordvecs.txt)
        - [`wordvecs.vcb`](https://inotwant-picture-public.oss-cn-beijing.aliyuncs.com/%E6%AF%95%E8%AE%BE%E7%9B%B8%E5%85%B3/%E6%9D%90%E6%96%99/embedding/wordvecs.vcb)
    - 已训练模型( models 文件夹下)
        - [`BiLSTM-CRF`](https://inotwant-picture-public.oss-cn-beijing.aliyuncs.com/%E6%AF%95%E8%AE%BE%E7%9B%B8%E5%85%B3/%E6%9D%90%E6%96%99/model/params_pass_00024.tar.gz)
        - [`mLSTM & CRF`](https://inotwant-picture-public.oss-cn-beijing.aliyuncs.com/%E6%AF%95%E8%AE%BE%E7%9B%B8%E5%85%B3/%E6%9D%90%E6%96%99/model/params_pass_00023.tar.gz)
3. 将项目代码导入到 `Docker` 中
    - 开启 `PaddlePaddle` 相关容器
    > `sudo docker run -d --name paddle -p 2202:22 paddle` (这里镜像名称为 `paddle` , 并且容器命名为 `paddle`)  
    > `sudo docker exec paddle /etc/init.d/ssh start` (启动容器中的 `ssh` , 若未使用 `pycharm` 此步骤不需要)  
    - 将代码拷贝到 `Docker` 中
    > 方法一: `sudo docker cp QA paddle:/home/` (这里将 QA 拷贝到容器中的 home 文件夹下)  
    > 方法二: 可使用 `pycharm` 进行可视化操作,这里不再详细描述

## 演示
- 使用 `BiLSTM-CRF` 模型
> `sudo docker exec paddle /usr/bin/python /home/QA/src/application.py /home/QA/data/qe_text` (`qe_text` 中包含了 **查询** 和 **证据文章** , 下面对它的结构做详细介绍)
- 使用 `Match-LSTM & CRF` 模型
> `sudo docker exec paddle /usr/bin/python /home/QA/src/mLSTM_crf/mLSTM_crf_application.py /home/QA/data/qe_text`
- 说明
> 以上运行后将会得到一串标识,这些标识的数量等于证据文章中词的个数.其中, `0;` 标识对应就是答案.
- `qe_text` 结构
> 第一行为分词后的 "查询" .  
> 接下来的所有都是分词后的 "证据文章". 每一行代表一篇.

## 模型
- 训练 `BiLSTM-CRF`
> 1. 所有的参数设置都在 `config.py` 中,请自行修改.(注意,这里的训练集来自于 `WebQA` ,请自行下载. `WebQA` 是由百度发布的,现在好像不提供下载了,如需要邮箱联系);  
> 2. 运行 `train.py` 即可进行训练;
- 训练 `Match-LSTM & CRF`
> 类似上述,略!

## 参考
- [Neural_qa](https://github.com/PaddlePaddle/models/tree/develop/neural_qa)
