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

from pathlib import Path
proxy_project_path = Path("/home/workspace/music-content-ai-generate-proxy")
sys.path.append(str(proxy_project_path))
from services.status_callback import task_callback
from services.status_callback import upload_local_file_to_nos


# 创建FastAPI应用实例
app = FastAPI()

cosyvoice = CosyVoice2('/home/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
# cosyvoice = CosyVoice2('/home/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
spk_emb_dict = torch.load('/home/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/spk2embedding.pt', map_location='cpu')
# prompt_speech_16k = load_wav('./asset/spk12649899906_00157.wav', 16000)
prompt_speech_16k = load_wav('./asset/spk302346072_00060.wav', 16000)

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
    tts_text = data['text']
    spk_id = data['speakerId']
    task_id = data['taskId']
    is_sync = data['isSync']
    results = list(cosyvoice.inference_sft_peng(
        tts_text, spk_id, prompt_speech_16k, spk_emb_dict[spk_id], stream=False
    ))
    if results:
        all_audio = torch.cat([j['tts_speech'] for j in results], dim=-1)
        output_dir = os.path.join(os.getcwd(), "output")
        filename = f"sft_instruct_{str(uuid.uuid4()).replace('-', '')}.wav"
        file_path = os.path.join(output_dir, filename)
        torchaudio.save(file_path, all_audio, cosyvoice.sample_rate)
        file_abs_path = os.path.abspath(file_path)
        # 获取端口号
        port = app.state.port if hasattr(app.state, 'port') else None
        execution_time = (time.time() - start_time) * 1000
        logging.info(f"vllm tts success, task_id: {task_id}, tts text: {tts_text}, execution time: {execution_time}")
        if is_sync:
            upload_result = upload_local_file_to_nos(file_abs_path)
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
    parser.add_argument("--port", type=int, help="服务端口", required=True)
    args = parser.parse_args()

    # 保存端口到app.state，方便后续调用
    app.state.port = args.port

    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port
    )