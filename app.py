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

from starlette.middleware.cors import CORSMiddleware

# 환경 변수 설정
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
app = FastAPI(
    title="Novel Generation API",
    version="0.1.0",
    description="Generate novels from images using GPT-4"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 이미지 캡션 생성 함수
def get_image_captioning(image: str) -> str:
    """이미지를 GPT-4 모델로 처리하여 캡션 생성"""
    response = openai.chat.completions.create(
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
    print('image 캡션 시작')
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
        caption = get_image_captioning(base64_image)
        output_dict[idx] = caption
    print('image 캡션 완료')
    return output_dict

# 캡션을 기반으로 소설 생성
def caption_to_novel(caption_dict: Dict[int, str], name: str, genre: str) -> str:
    """캡션으로부터 소설 생성"""
    print('novel 생성 시작')
    caption_corpus = " ".join([f"{key}: {value}" for key, value in caption_dict.items()])

    prompt = (
        f"""당신은 창의적이고 이야기 제작에 능숙한 AI 작가입니다. 
        사용자의 입력과 사진의 캡션을 기반으로 매력적이고 장르에 맞는 스토리를 만들어 주세요. 
        아래는 스토리를 작성하기 위한 조건입니다:

        입력 정보:
        name: {name}
        genre: {genre}
        사진 캡션 정보:
        {caption_corpus}
        
        요구 사항:
        사용자의 이름과 선택한 장르를 반영하여 스토리를 만드세요.
        사진 캡션은 스토리 속에서 자연스럽게 녹아들도록 하되, 각 캡션이 끝날 때 [image]를 추가하여 사진 위치를 표시하세요.
        스토리는 논리적이고 흥미롭게 진행되어야 합니다.
        
        입력 예시:
        name: 민준
        genre: 판타지
        사진 캡션: ["숲 속의 고대 유적", "마법의 빛을 발하는 수정 구슬", "전설의 용과의 조우"]
        출력 예시:
        민준은 어릴 적부터 고대 유적을 탐험하는 꿈을 꿨다. 어느 날, 그는 신비로운 숲으로 떠나기로 결심했다. 숲 깊은 곳에서 그는 오랜 시간 잊혀졌던 고대 유적을 발견했다. 유적은 초록 이끼로 뒤덮여 있었지만, 그곳에 서린 고대의 힘은 여전히 강렬했다. [image]
        그가 유적 안으로 들어서자, 빛나는 수정 구슬이 공중에 떠올라 그의 주위를 감쌌다. 구슬은 그의 손길에 반응하며 은은한 빛을 내뿜었다. 이는 분명히 고대 마법의 유물임에 틀림없었다. [image]
        구슬의 빛을 따라가던 민준은 전설로만 전해지던 용과 마주쳤다. 용은 그의 방문을 기다렸다는 듯이 웅장한 목소리로 말했다. "드디어, 내가 선택한 인간이 왔구나." [image]
        이제 민준의 모험은 막 시작되었다.
        """
    )

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 뛰어난 소설가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    print('novel 생성 완료')

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
    print(f"{name}이 선택한 {genre.value} 장르")
    captions = await image_to_caption(images)  # 이미지에서 캡션 생성
    print(f'')
    novel = caption_to_novel(captions, name, genre)  # 캡션을 기반으로 소설 생성
    return NovelResponse(captions=captions, novel=novel)

# FastAPI 애플리케이션 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)
