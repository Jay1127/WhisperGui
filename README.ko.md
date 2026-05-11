# Whisper GUI

Whisper 음성 변환 옵션을 테스트하기 위한 간단한 로컬 데스크톱 앱이다.

## 실행 준비

먼저 가상 환경을 만들고 활성화한다.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Whisper 로컬 실행에는 FFmpeg가 필요하다.

```powershell
choco install ffmpeg
```

또는 Scoop을 사용한다면 다음처럼 설치할 수 있다.

```powershell
scoop install ffmpeg
```

## 설치

Whisper 패키지를 설치한다.

```powershell
pip install openai-whisper
```

또는 `requirements.txt`로 설치한다.

```powershell
pip install -r requirements.txt
```

## 실행

```powershell
python app.py
```

## 테스트 항목

- 로컬 Whisper로 음성 파일 텍스트 변환
- `transcribe`, `translate` 작업 선택
- 로컬 모델 크기 선택
- 출력 형식 선택
- 언어 지정
- 프롬프트 지정
- 변환 진행률 표시
- 단어 타임스탬프 생성
- 자막 줄 길이와 단어 수 옵션
- 자막 구두점 옵션
- 디코딩 관련 옵션

`Highlight words`, 자막 줄 옵션, 문장부호 옵션, 무음 건너뛰기 기준은 `Word timestamps`를 켰을 때만 사용할 수 있다.

`Output format`을 `all`로 선택하면 선택한 출력 파일 하나가 아니라 입력 파일 이름을 기준으로 여러 출력 파일이 같은 폴더에 생성된다.

앱은 변환된 텍스트를 선택한 출력 경로 또는 폴더에 저장하고, 창 안에서 미리보기를 보여준다.
