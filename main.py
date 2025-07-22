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


# 创建FastAPI应用实例
app = FastAPI()

# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
spk_emb_dict = torch.load('miaoshi_spk2embedding.pt', map_location='cpu')
prompt_speech_16k = load_wav('./asset/spk12649899906_00157.wav', 16000)

# 定义处理POST请求的接口
@app.post("/vllm/tts")
async def vllm_tts(request: Request):
    data = await request.json()
    tts_text = data['text']
    spk_id = data['speakerId']
    results = list(cosyvoice.inference_sft_peng(
        tts_text, spk_id, prompt_speech_16k, spk_emb_dict[spk_id], stream=False
    ))
    if results:
        all_audio = torch.cat([j['tts_speech'] for j in results], dim=-1)
        filename = f"sft_instruct_{str(uuid.uuid4()).replace('-', '')}.wav"
        torchaudio.save(filename, all_audio, cosyvoice.sample_rate)
        abs_path = os.path.abspath(filename)
        logging.info(f"音频已保存，绝对路径为: {abs_path}")
    logging.info(f"tts success, tts text: {tts_text}")
    return {
        "status": "success",
        "file_path": abs_path if results else None
    }

# 主函数，用来创建应用实例并运行
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="服务端口", required=True)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port
    )