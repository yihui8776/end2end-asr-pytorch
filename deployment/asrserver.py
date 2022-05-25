import argparse
import base64
import json
from flask import Flask, Response, request

import os
import wave
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from utils.functions import save_model, load_model
from utils.lstm_utils import LM
from models.asr.transformer import Transformer, Encoder, Decoder
from utils import constant
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer, calculate_cer_en_zh

# from utils.ops import decode_wav_bytes
# import ltwavelib



API_STATUS_CODE_OK = 20000  # OK
API_STATUS_CODE_CLIENT_ERROR = 30000
API_STATUS_CODE_CLIENT_ERROR_FORMAT = 30001  # 请求数据格式错误
API_STATUS_CODE_CLIENT_ERROR_CONFIG = 30002  # 请求数据配置不支持
API_STATUS_CODE_SERVER_ERROR = 40000
API_STATUS_CODE_SERVER_ERROR_RUNNING = 40001  # 服务器运行中出错

parser = argparse.ArgumentParser(description='ASRT HTTP+Json RESTful API Service')
parser.add_argument('--listen', default='0.0.0.0', type=str, help='the network to listen')
parser.add_argument('--port', default='20001', type=str, help='the port to listen')
args = parser.parse_args()

app = Flask("ASRT API Service")

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
# 默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
OUTPUT_SIZE = 1428


def decode_wav_bytes(samples_data: bytes, channels: int = 1, byte_width: int = 2) -> list:
    '''
    解码wav格式样本点字节流，得到numpy数组
    '''
    numpy_type = np.short
    if byte_width == 4:
        numpy_type = np.int
    elif byte_width != 2:
        raise Exception('error: unsurpport byte width `' + str(byte_width) + '`')
    wave_data = np.fromstring(samples_data, dtype=numpy_type)  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, channels  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置
    return wave_data

#这里调用的是model的evaluate，后续可以再优化，直接predict
def evaluate(model, test_loader, lm=None):
    """
    Evaluation
    args:
        model: Model object
        test_loader: DataLoader object
    """
    model.eval()

    total_word, total_char, total_cer, total_wer = 0, 0, 0, 0
    total_en_cer, total_zh_cer, total_en_char, total_zh_char = 0, 0, 0, 0

    with torch.no_grad():
        test_pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
        for i, (data) in enumerate(test_pbar):
            src, tgt, src_percentages, src_lengths, tgt_lengths = data

            if constant.USE_CUDA:
                src = src.cuda()
                tgt = tgt.cuda()

            batch_ids_hyps, batch_strs_hyps, batch_strs_gold = model.evaluate(
                src, src_lengths, tgt, beam_search=constant.args.beam_search, beam_width=constant.args.beam_width,
                beam_nbest=constant.args.beam_nbest, lm=lm, lm_rescoring=constant.args.lm_rescoring,
                lm_weight=constant.args.lm_weight, c_weight=constant.args.c_weight, verbose=constant.args.verbose)

            for x in range(len(batch_strs_gold)):
                hyp = batch_strs_hyps[x].replace(constant.EOS_CHAR, "").replace(constant.SOS_CHAR, "").replace(
                    constant.PAD_CHAR, "")

            return hyp


# load_path = "save/aishell_drop0.1_cnn_batch12_4_vgg_layer4/best_model.th"
load_path = "save/aishell_drop0.1_cnn_batch12_4_vgg_layer4/epoch_516.th"
model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(load_path)

audio_conf = dict(sample_rate=loaded_args.sample_rate,
                  window_size=loaded_args.window_size,
                  window_stride=loaded_args.window_stride,
                  window=loaded_args.window,
                  noise_dir=loaded_args.noise_dir,
                  noise_prob=loaded_args.noise_prob,
                  noise_levels=(loaded_args.noise_min, loaded_args.noise_max))

test_manifest_path = ["manifests/testapi.csv"]

test_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath_list=test_manifest_path, label2id=label2id,
                               normalize=True, augment=False)

test_sampler = BucketingSampler(test_data, batch_size=1)
test_loader = AudioDataLoader(test_data, num_workers=2, batch_sampler=test_sampler)

# print('语音识别最终结果：\n',res)

lm = None

if constant.args.lm_rescoring:
    lm = LM(constant.args.lm_path)


class AsrtApiResponse:
    '''
    ASRT语音识别基于HTTP协议的API接口响应类
    '''

    def __init__(self, status_code, status_message='', result=''):
        self.status_code = status_code
        self.status_message = status_message
        self.result = result

    def to_json(self):
        '''
        类转json
        '''
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True)


# api接口根url:GET
@app.route('/', methods=["GET"])
def index_get():
    '''
    根路径handle GET方法
    '''
    buffer = ''
    with open('assets/default.html', 'r', encoding='utf-8') as file_handle:
        buffer = file_handle.read()
    return Response(buffer, mimetype='text/html; charset=utf-8')


# api接口根url:POST
@app.route('/', methods=["POST"])
def index_post():
    '''
    根路径handle POST方法
    '''
    json_data = AsrtApiResponse(API_STATUS_CODE_OK, 'ok')
    buffer = json_data.to_json()
    return Response(buffer, mimetype='application/json')


# 获取分类列表
@app.route('/<level>', methods=["POST"])
def recognition_post(level):
    '''
    其他路径 POST方法
    '''
    # 读取json文件内容
    try:
        if level == 'speech':
            request_data = request.get_json()
            samples = request_data['samples']
            # 量化位数(采样大小，采样宽度):波每一个时刻都有一个对应的能量值，在计算机中用整数存储。通常使用16bit有符号整数存储，采样大小是16bit
            wavdata_bytes = base64.urlsafe_b64decode(bytes(samples, encoding='utf-8'))
            sample_rate = request_data['sample_rate']
            channels = request_data['channels']  # 声道数
            byte_width = request_data['byte_width']

            # wavdata = decode_wav_bytes(samples_data=wavdata_bytes,
            #                       channels=channels, byte_width=byte_width)

            # lw = ltwavelib.LtWave()
            audio_conf = dict(sample_rate=sample_rate,
                              window_size=loaded_args.window_size,
                              window_stride=loaded_args.window_stride,
                              window=loaded_args.window,
                              noise_dir=loaded_args.noise_dir,
                              noise_prob=loaded_args.noise_prob,
                              noise_levels=(loaded_args.noise_min, loaded_args.noise_max))

            output_dir = "./mediacache"
            output_path = output_dir + "/test.wav"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # save the recorded data as wav file using python `wave` module
            # 通过fromstring函数将字符串转换为列表，通过其参数dtype指定转换后的数据格式，由于我们的声音格式是以两个字节表示一个取样值，因此采用short数据类型转换
            # wave_data = np.fromstring(wavdata_bytes, dtype=np.short)  #字符串格式读入 ，
            wave_data = np.frombuffer(wavdata_bytes,
                                      dtype='int16')  ##frombuffer将data以流的形式读入转化成ndarray对象 ，将波形数据转换为列表【矩阵】
            # print(wavdata_bytes)
            wave_data.shape = -1, channels  # 通常是-1，2 声道数
            # -1  表示行数未知；   2 表示2列
            # 声音文件是双声道的，因此它由左右两个声道的取样交替构成：LRLRLRLR....LR（L表示左声道的取样值，R表示右声道取样值）。修改wave_data的sharp之后：
            # [[0 0]  [0 0]  ...  [0 0]  [0 0]  [0 0]]
            temp_data = wave_data
            temp_data.shape = 1, -1  # 将其转置--行列转换
            temp_data = temp_data.astype(np.short)
            wf = wave.open(output_path, 'wb')
            # 配置声道数、量化位数和取样频率
            wf.setnchannels(channels)
            wf.setsampwidth(byte_width)
            wf.setframerate(sample_rate)
            # 将wav_data转换为二进制数据写入文件
            wf.writeframes(temp_data.tobytes())
            wf.close()

            # print(audio_conf)
            result = evaluate(model, test_loader, lm=lm)
            json_data = AsrtApiResponse(API_STATUS_CODE_OK, 'speech level')
            json_data.result = result
            buffer = json_data.to_json()
            print('output:', buffer)
            return Response(buffer, mimetype='application/json')
        elif level == 'language':
            request_data = request.get_json()

            seq_pinyin = request_data['sequence_pinyin']

            # result = ml.SpeechToText(seq_pinyin)
            result = 'pinyin'
            json_data = AsrtApiResponse(API_STATUS_CODE_OK, 'language level')
            json_data.result = result
            buffer = json_data.to_json()
            print('output:', buffer)
            return Response(buffer, mimetype='application/json')
        elif level == 'all':
            request_data = request.get_json()

            samples = request_data['samples']
            wavdata_bytes = base64.urlsafe_b64decode(samples)
            sample_rate = request_data['sample_rate']
            channels = request_data['channels']
            byte_width = request_data['byte_width']

            wavdata = decode_wav_bytes(samples_data=wavdata_bytes,
                                       channels=channels, byte_width=byte_width)
            # result_speech = ms.recognize_speech(wavdata, sample_rate)
            # result = ml.SpeechToText(result_speech)
            result = 'all level'

            json_data = AsrtApiResponse(API_STATUS_CODE_OK, 'all level')
            json_data.result = result
            buffer = json_data.to_json()
            print('ASRT Result:', result, 'output:', buffer)
            return Response(buffer, mimetype='application/json')
        else:
            request_data = request.get_json()
            print('input:', request_data)
            json_data = AsrtApiResponse(API_STATUS_CODE_CLIENT_ERROR, '')
            buffer = json_data.to_json()
            print('output:', buffer)
            return Response(buffer, mimetype='application/json')
    except Exception as except_general:
        request_data = request.get_json()
        # print(request_data['sample_rate'], request_data['channels'],
        # request_data['byte_width'], len(request_data['samples']),
        # request_data['samples'][-100:])
        json_data = AsrtApiResponse(API_STATUS_CODE_SERVER_ERROR, str(except_general))
        buffer = json_data.to_json()
        # print("input:", request_data, "\n", "output:", buffer)
        print("output:", buffer, "error:", except_general)
        return Response(buffer, mimetype='application/json')


if __name__ == '__main__':
    # for development env
    app.run(host='0.0.0.0', port=20001)
    # for production env
    # import waitress
    # waitress.serve(app, host=args.listen, port=args.port)
