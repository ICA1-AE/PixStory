# 250109, 250110 


import openai
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. image에서 caption을 생성하는 코드
# input : image 경로 list
# output : {image number:caption} 
def image_to_caption(image_list) :
  output_dict = {}
  for idx, image_path in enumerate(image_list) : 
    # 1) 이미지 캡셔닝 모델 로드
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # 2) 이미지 로드 및 resize, captioning 
    image = Image.open(image_path).convert("RGB").resize((224, 224))

    inputs = caption_processor(images=image, return_tensors="pt")
    caption_output = caption_model.generate(**inputs)
    caption = caption_processor.decode(caption_output[0], skip_special_tokens=True)

    output_dict[idx] = caption
  return output_dict

# 2. caption에서 을 생성하는 코드드
def caption_to_novel(caption_dict, name):
    # Step 1: caption_dict의 내용을 하나로 병합
    caption_corpus = " ".join([f"{key}: {value}" for key, value in caption_dict.items()])
    
    # Step 2: GPT-4에게 소설 생성 요청
    prompt = (
    f"다음은 사용자가 제공한 여러 이미지에 대한 설명입니다. 주인공의 이름은 '{name}'입니다. "
    "각 이미지는 특정 시점의 과거를 나타내며, 입력된 이미지의 수에 따라 시간 순서대로 서술해 주세요. "
    "각각의 이미지는 다음과 같은 구조 중 하나에 맞춰 설명됩니다:\n"
    "- 어린 시절의 장면: 밝고 순수한 기억\n"
    "- 성장기의 장면: 변화와 도전\n"
    "- 현재를 암시하는 과거의 한 장면: 반성이나 성찰\n\n"
    f"이미지 설명: {caption_corpus}\n\n"
    f"'{name}'의 이야기를 입력된 모든 이미지에 대해 시간 순으로 연결하며, 감정적이고 자연스러운 한국어 문장으로 표현해 주세요. "
    "각 설명은 1~2문장으로 간결하게 작성해 주세요."
)

    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 뛰어난 소설가입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    
    # GPT-4의 응답 내용
    novel_text = response['choices'][0]['message']['content']

    return novel_text





# 예제 입력
name = "민수" 
image_list= ["D:/Notwork/INNER_CIRCLE/ICAI/ex_jjang.jpg"]


caption_dict = image_to_caption(image_list)
novel_text = caption_to_novel(caption_dict, name)
print(novel_text)
