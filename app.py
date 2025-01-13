from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict
import openai
from PIL import Image
from dotenv import load_dotenv
import os
import io
import base64
import uvicorn
from enum import Enum

# 환경 변수 설정
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
app = FastAPI(
    title="Novel Generation API",
    version="0.1.0",
    description="Generate novels from images using GPT-4"
)

# 이미지 캡션 생성 함수
async def get_image_captioning(image: str) -> str:
    """이미지를 GPT-4 모델로 처리하여 캡션 생성"""
    response = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """주어진 이미지를 기반으로 간결하고 명확한 디스크립션을 생성합니다. 
                        디스크립션에는 다음 요소들을 포함해야 합니다:

                        주요 객체: 사진에서 가장 눈에 띄는 물체, 인물 또는 요소.
                        배경 정보: 배경에서 보이는 주요 환경, 장소, 또는 색상 조합.
                        활동 또는 상황: 사진 속 인물이나 물체가 수행 중인 활동 또는 상태.
                        분위기와 감정: 사진이 전달하는 전반적인 분위기(예: 평화로운, 역동적인, 따뜻한).
                        디스크립션은 한 문단으로 작성하며, 생동감 있는 언어를 사용하여 사진의 이미지를 쉽게 떠올릴 수 있도록 작성해주세요.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                ],
            }
        ],
    )
    return response.choices[0].message.content

# 여러 이미지에서 캡션 생성
async def image_to_caption(images: List[UploadFile]) -> Dict[int, str]:
    """이미지 리스트에서 캡션을 생성"""
    output_dict = {}
    for idx, image_file in enumerate(images):
        # 이미지 읽기 및 전처리
        content = await image_file.read()
        image = Image.open(io.BytesIO(content)).resize((224, 224)).convert("RGB")

        # 이미지를 base64로 인코딩
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 캡션 생성
        caption = await get_image_captioning(base64_image)
        output_dict[idx] = caption

    return output_dict

# 캡션을 기반으로 소설 생성
async def caption_to_novel(caption_dict: Dict[int, str], name: str, genre: str) -> str:
    """캡션으로부터 소설 생성"""
    caption_corpus = " ".join([f"{key}: {value}" for key, value in caption_dict.items()])

    prompt = (
        f"다음은 사용자가 제공한 여러 이미지에 대한 설명입니다. 주인공의 이름은 '{name}'입니다. "
        f"장르는 '{genre}'입니다. "
        "각 이미지는 특정 시점의 과거를 나타내며, 입력된 이미지의 수에 따라 시간 순서대로 서술해 주세요. "
        "각각의 이미지는 다음과 같은 구조 중 하나에 맞춰 설명됩니다:\n"
        "- 어린 시절의 장면: 밝고 순수한 기억\n"
        "- 성장기의 장면: 변화와 도전\n"
        "- 현재를 암시하는 과거의 한 장면: 반성이나 성찰\n\n"
        f"이미지 설명: {caption_corpus}\n\n"
        f"'{name}'의 이야기를 입력된 모든 이미지에 대해 시간 순으로 연결하며, "
        "감정적이고 자연스러운 한국어 문장으로 표현해 주세요. "
        "누구나 빠져들게끔 재밌게 만들어주세요. "
        "각 문단은 500자를 넘지 않도록 해주세요."
    )

    response = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 뛰어난 소설가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

# 선택 가능한 장르 Enum 정의
class Genre(str, Enum):
    fantasy = "판타지"
    sf = "SF"
    romance = "로맨스"
    mystery = "미스터리"
    adventure = "모험"

# Pydantic 모델 정의
class NovelRequest(BaseModel):
    name: str
    genre: Genre

class NovelResponse(BaseModel):
    captions: Dict[int, str]
    novel: str

# 소설 생성 API 엔드포인트
@app.post(
    "/generate/",
    response_model=NovelResponse,
    tags=["novel"],
    summary="Generate novel from images",
    description="Upload images and get a generated novel based on image captions"
)
async def generate_novel(
    name: str,
    genre: Genre,
    images: List[UploadFile] = File(...),
):
    """
    이미지들과 이름을 입력받아 소설을 생성합니다.

    - **name**: 소설 속 주인공의 이름
    - **genre**: 소설 장르 (예: 판타지, SF, 로맨스, 미스터리, 모험)
    - **images**: 시간 순서대로 정렬된 이미지 파일들
    """
    captions = await image_to_caption(images)  # 이미지에서 캡션 생성
    novel = await caption_to_novel(captions, name, genre)  # 캡션을 기반으로 소설 생성
    return NovelResponse(captions=captions, novel=novel)

# FastAPI 애플리케이션 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)
