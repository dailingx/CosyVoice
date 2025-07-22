import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch




to_syn_text = []
for line in open('spk12649899906_baixue_text.txt', encoding='utf-8'):
    line = line.strip()
    line_elems = line.split()
    utt_id = line_elems[0]
    text = ' '.join(line_elems[1:])
    to_syn_text.append((utt_id, text))





cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)


prompt_speech_16k = load_wav('./asset/spk12649899906_00157.wav', 16000)


# for utt_id, syn_text in to_syn_text:
#     for i, j in enumerate(cosyvoice.inference_zero_shot(syn_text, '这是我说的嘛，啊？回消息按天来算的，你下次就可以按周来算。', prompt_speech_16k, stream=False)):
#         torchaudio.save(f'{utt_id}_zero_shot_{i:02}.wav', j['tts_speech'], cosyvoice.sample_rate)






spk_emb_dict = torch.load('pretrained_models/CosyVoice2-0.5Bspk2embedding.pt', map_location='cpu')
spk_id = 'spk12649899906'


for utt_id, syn_text in to_syn_text:
    for i, j in enumerate(cosyvoice.inference_sft_peng(syn_text, spk_id, prompt_speech_16k, spk_emb_dict[spk_id], stream=False)):
        torchaudio.save(f'{utt_id}_sft_instruct_{i:02}.wav', j['tts_speech'], cosyvoice.sample_rate)

