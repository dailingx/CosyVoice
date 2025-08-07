from fastapi import FastAPI, HTTPException
from fastapi import Request
import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
import torchaudio
import torch
import uuid
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import logging
import argparse
import os
import time
import re

from pathlib import Path
proxy_project_path = Path("/home/workspace/music-content-ai-generate-proxy")
sys.path.append(str(proxy_project_path))
from services.status_callback import task_callback
from services.status_callback import upload_local_file_to_nos, download_file_from_nos


# 创建FastAPI应用实例
app = FastAPI()

# 全局变量声明
cosyvoice = None
spk_emb_dict = {}
# 添加spk_id到prompt_speech_16k的缓存字典
spk_prompt_cache = {}


def init_spk_cache():
    # prompt_speech_16k = load_wav('./asset/spk12649899906_00157.wav', 16000)
    preheat_prompt_speech_16k = load_wav('./asset/spk302346072_00060.wav', 16000)
    spk_prompt_cache['spk302346072'] = preheat_prompt_speech_16k


def initialize_cosyvoice(no_vllm=False):
    """初始化 CosyVoice2 实例"""
    global cosyvoice, spk_emb_dict, prompt_speech_16k
    
    # 根据参数决定是否反转布尔值
    load_jit = True
    load_trt = True
    load_vllm = True
    fp16 = True
    
    if no_vllm:
        load_jit = not load_jit
        load_trt = not load_trt
        load_vllm = not load_vllm
        fp16 = not fp16
    
    cosyvoice = CosyVoice2('/home/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B', 
                          load_jit=load_jit, load_trt=load_trt, load_vllm=load_vllm, fp16=fp16)
    spk_emb_dict = torch.load('/home/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/spk2embedding.pt', map_location='cpu')
    init_spk_cache()
    logging.info(f"initialize cosyvoice2 success, use_vllm: {not no_vllm}")


def preprocess_text_input(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+|\s+(?=[\u4e00-\u9fff])', '，', text)
    text = re.sub(r'~+', '，', text)
    text = re.sub(r'、+', '，', text)
    text = re.sub(r'，+', '，', text)
    text = re.sub(r'[，,、]+$', '。', text)
    return text


def async_task_callback(task_id, success, file_path, execution_time, port):
    try:
        status = "TASK_SUCCESS" if success else 'TASK_FAIL'
        data = {
            "outputs": {
                "audio": [
                    {
                        "file_path": file_path
                    }
                ]
            },
            "status": {
                "status_str": "success"
            }
        }
        gpu_uuid = task_callback(task_id, status, data, execution_time, port)
        logging.info(f"Task Callback completed, task_id: {task_id}, status: {status}, gpu_uuid: {gpu_uuid}")
    except Exception as e:
        logging.error(f"Traceback status callback error when task finish, task_id: {task_id}, e: {str(e)}")


# 定义处理POST请求的接口
@app.post("/vllm/tts")
async def vllm_tts(request: Request):
    start_time = time.time()
    data = await request.json()
    tts_text = preprocess_text_input(data['text'])
    spk_id = data['speakerId']
    task_id = data['taskId']
    is_sync = data['isSync']
    spk_speech_nos = data.get('speakerSpeechNosKey', None)
    output_nos_endpoint = data.get('outputNosEndpoint', None)
    output_nos_bucket = data.get('outputNosBucket', None)
    output_file_format = data.get('outputFileFormat', "wav")

    # 检查spk_id是否在缓存中
    if spk_id in spk_prompt_cache:
        prompt_speech_16k = spk_prompt_cache[spk_id]
    else:
        logging.info(f"vllm tts: spk_id {spk_id} 未缓存，将下载并缓存prompt_speech_16k")
        spk_prompt_speech_filename = spk_speech_nos.split('/')[-1]
        if '.' not in spk_prompt_speech_filename:
            spk_prompt_speech_filename += '.wav'
        spk_prompt_speech_path = os.path.join('./asset', f"{spk_id}_{spk_prompt_speech_filename}")
        download_success = download_file_from_nos(nos_key=spk_speech_nos, save_path=spk_prompt_speech_path)
        if download_success is not True:
            return {
                "status": "fail",
                "message": "download audio file from nos fail!"
            }
        prompt_speech_16k = load_wav(spk_prompt_speech_path, 16000)
        # 将spk_id和对应的prompt_speech_16k添加到缓存中
        spk_prompt_cache[spk_id] = prompt_speech_16k
        logging.info(f"vllm tts: spk_id {spk_id} 已添加到缓存")
    
    results = list(cosyvoice.inference_sft_peng(
        tts_text, spk_id, prompt_speech_16k, spk_emb_dict[spk_id], stream=False
    ))
    if results:
        all_audio = torch.cat([j['tts_speech'] for j in results], dim=-1)
        output_dir = os.path.join(os.getcwd(), "output")
        filename = f"sft_instruct_{spk_id}_{str(uuid.uuid4()).replace('-', '')}.{output_file_format}"
        file_path = os.path.join(output_dir, filename)
        torchaudio.save(file_path, all_audio, cosyvoice.sample_rate)
        file_abs_path = os.path.abspath(file_path)
        # 获取端口号
        port = app.state.port if hasattr(app.state, 'port') else None
        execution_time = (time.time() - start_time) * 1000
        logging.info(f"vllm sft instruct success, task_id: {task_id}, tts text: {tts_text}, execution time: {execution_time}")
        if is_sync:
            upload_result = upload_local_file_to_nos(file_abs_path, output_nos_endpoint, output_nos_bucket)
            return {
                "status": "success",
                "fileNos": upload_result['fileNos'],
                "duration": upload_result['duration'],
                "executionTime": execution_time
            }
        else:
            async_task_callback(task_id, True, file_abs_path, 0, port)
            return {
                "status": "success",
            }


@app.post("/vllm/zero/shot")
async def vllm_zero_shot(request: Request):
    start_time = time.time()
    data = await request.json()
    tts_text = preprocess_text_input(data['text'])
    spk_id = data['speakerId']
    spk_speech_nos = data['speakerSpeechNosKey']
    spk_text = data['speakerText']
    task_id = data['taskId']
    is_sync = data['isSync']
    output_nos_endpoint = data.get('outputNosEndpoint', None)
    output_nos_bucket = data.get('outputNosBucket', None)
    output_file_format = data.get('outputFileFormat', "wav")
    # 判断spk_id是否在spk2info中
    if spk_id in cosyvoice.frontend.spk2info:
        results = list(cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=spk_id, stream=False))
    else:
        logging.info(f"vllm zero shot: spk_id {spk_id} 未缓存，将添加到缓存")
        spk_prompt_speech_filename = spk_speech_nos.split('/')[-1]
        if '.' not in spk_prompt_speech_filename:
            spk_prompt_speech_filename += '.wav'
        spk_prompt_speech_path = os.path.join('./asset', f"{spk_id}_{spk_prompt_speech_filename}")
        download_success = download_file_from_nos(nos_key=spk_speech_nos, save_path=spk_prompt_speech_path)
        if download_success is not True:
            return {
                "status": "fail",
                "message": "download audio file from nos fail!"
            }
        spk_prompt_speech_16k = load_wav(spk_prompt_speech_path, 16000)
        cosyvoice.add_zero_shot_spk(spk_text, spk_prompt_speech_16k, spk_id)
        results = list(cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=spk_id, stream=False))
        cosyvoice.save_spkinfo()
        logging.info(f"vllm zero shot: spk_id {spk_id} 已添加到缓存")
    if results:
        all_audio = torch.cat([j['tts_speech'] for j in results], dim=-1)
        output_dir = os.path.join(os.getcwd(), "output")
        filename = f"zero_shot_{spk_id}_{str(uuid.uuid4()).replace('-', '')}.{output_file_format}"
        file_path = os.path.join(output_dir, filename)
        torchaudio.save(file_path, all_audio, cosyvoice.sample_rate)
        file_abs_path = os.path.abspath(file_path)
        # 获取端口号
        port = app.state.port if hasattr(app.state, 'port') else None
        execution_time = (time.time() - start_time) * 1000
        logging.info(f"vllm zero shot success, task_id: {task_id}, tts text: {tts_text}, execution time: {execution_time}")
        if is_sync:
            upload_result = upload_local_file_to_nos(file_abs_path, output_nos_endpoint, output_nos_bucket)
            return {
                "status": "success",
                "fileNos": upload_result['fileNos'],
                "duration": upload_result['duration'],
                "executionTime": execution_time
            }
        else:
            async_task_callback(task_id, True, file_abs_path, 0, port)
            return {
                "status": "success",
            }


# 主函数，用来创建应用实例并运行
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8190, help="服务端口")
    parser.add_argument("--no_vllm", action="store_true", help="不使用vLLM")
    args = parser.parse_args()

    # 保存端口到app.state，方便后续调用
    app.state.port = args.port

    # 初始化 CosyVoice2 实例
    initialize_cosyvoice(args.no_vllm)

    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port
    )