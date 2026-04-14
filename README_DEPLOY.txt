3일 내 10% 급등 확률 스크리너 배포용 구조

이 폴더는 어제 만든 앱과 연결되지 않는 별도 배포용 구조입니다.
즉, 새 저장소 / 새 앱으로 따로 배포하면 완전히 분리해서 사용할 수 있습니다.

포함 파일
- streamlit_app.py : 배포용 시작 파일
- requirements.txt : 필요한 패키지 목록
- .streamlit/config.toml : Streamlit 실행 설정
- README_DEPLOY.txt : 배포 안내

배포 방법
1. 이 폴더 전체를 새 GitHub 저장소에 올립니다.
2. Streamlit Community Cloud에서 새 앱을 만듭니다.
3. Repository는 새 저장소를 선택합니다.
4. Branch는 main
5. Main file path는 streamlit_app.py
6. Deploy를 누릅니다.

중요
- 이 배포용 구조는 어제 만든 키움 앱과 연동되지 않습니다.
- 새 저장소 이름도 별도로 만드는 것을 권장합니다.
  예: three-day-spike-screener
- 별도 앱 URL이 생성되므로 기존 앱과 완전히 분리됩니다.
