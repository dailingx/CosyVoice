from fastapi import FastAPI, HTTPException
from fastapi import Request
import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import logging
import argparse


# 创建FastAPI应用实例
app = FastAPI()

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)


# 定义处理POST请求的接口
@app.post("/vllm/tts")
async def vllm_tts(request: Request):
    # 获取原始请求的 JSON 数据
    body = await request.body()
    tts_text = body.text
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    set_all_random_seed(123)
    cosyvoice.inference_zero_shot(
            tts_text,
            '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)
    logging.info(f"tts success, tts text: {tts_text}")
    # 为了演示，这里直接返回处理结果
    return {
        "status": "success"
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