
### 各部分说明
- Dockerfile容器部署,基于http传输音频数据接口
- 服务端基于flask开发  ，app为asrserver.py
- getdevice.py 获取设备
- http客户端简单调用代码：  client_http.py
- 带界面客户端demo  ：recorddemo.py

#### 服务端
构建容器，并用run.sh 运行服务器容器服务

接口端口 20001 要开放

接口参数 ：

- channels ： 声道数
- sample_rate ：采样频率，每秒多少次采样
- sample_width ：采样宽度，量化位数
- samples:  数据

返回 status和结果

- 20000 ： 正确结果  OK
- 30000 :  客户端错误
- 30001 ： 请求数据格式错误
- 30002 ： 请求数据配置不支持
- 40000 ： API状态错误
- 40001 ： 服务器运行中出错

#### 客户端

读取本地音频wav文件，并获得各个参数，通过json传输,返回结果


界面demo

点击录制，打开本地麦克风 录音
录制完成可以点击播放
点击选择识别，  打开本地wav文件，并传递到服务端返回识别结果并展示

![image](https://github.com/yihui8776/end2end-asr-pytorch/blob/master/deployment/demo.jpg)



