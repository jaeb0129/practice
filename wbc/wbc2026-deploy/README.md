# WBC 2026 해설 도우미

## 구조
```
wbc2026-deploy/
├── api/
│   └── anthropic.js     ← Vercel Function (API 키 서버에서만 처리)
├── src/
│   ├── index.js
│   └── App.js           ← 메인 React 앱
├── public/
│   └── index.html
├── vercel.json
└── package.json
```

## Vercel 배포 (5분)

### 1. GitHub에 올리기
```bash
git init
git add .
git commit -m "init"
git remote add origin https://github.com/YOUR_ID/wbc2026.git
git push -u origin main
```

### 2. Vercel 연결
1. https://vercel.com → New Project
2. GitHub 저장소 import
3. Framework: **Create React App** (자동 감지)
4. **Environment Variables** 탭에서 추가:
   - Key: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-api03-...`
5. Deploy 클릭

### 3. 완료
배포 후 `https://wbc2026-xxx.vercel.app` 같은 URL이 생성됩니다.

## 로컬 개발
```bash
# 의존성 설치
npm install

# .env.local 생성
echo "ANTHROPIC_API_KEY=sk-ant-api03-..." > .env.local

# 개발 서버 실행 (Vercel CLI 사용 시 API 함수도 함께 실행)
npx vercel dev
# 또는 일반 React만
npm start
```

## 주의사항
- `ANTHROPIC_API_KEY`는 절대 클라이언트 코드에 넣지 마세요
- `api/anthropic.js`가 키를 서버에서만 사용하는 프록시 역할을 합니다
- MLB StatsAPI는 공개 API라 별도 키 불필요
