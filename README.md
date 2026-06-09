# SAFESEXUAL 🔒

> Unlearning for Sexual Content in Diffusion Models

## 배포 링크

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-배포됨-58a6ff?style=flat&logo=github)](https://kyungjunoh.github.io/SAFESEXUAL/index.html)

## 🧑‍🤝‍🧑 About Us

저희 팀은 성적으로 유해한 콘텐츠를 생성하는 확산 모델의 위험성을 줄이기 위한 연구를 수행합니다.
Machine Unlearning 기법을 활용해 모델의 특정 개념 생성 능력을 제거하는 방법을 탐구합니다.

| 이름 | 역할 |
| --- | --- |
| 오경준 | 프로젝트 총괄 · 모델 학습 및 실험 · GitHub Pages 배포 |
| 최정민 | 기획 및 선행 연구 조사 · UnSafeBench 데이터 분석 |
| 김윤수 | 실험 결과 정리 및 시각화 · 발표 자료 제작 |

`PyTorch` · `Diffusion Models` · `Machine Unlearning` · `PCA` · `Latent Space`

## 🚀 Project

텍스트-이미지 생성 모델(Diffusion Model)은 강력한 생성 능력을 갖추고 있지만,
그만큼 성적으로 유해한 콘텐츠를 생성할 위험도 존재합니다.
본 프로젝트는 **Machine Unlearning** 기법을 적용해 모델 재학습 없이(training-free) 또는
최소한의 학습으로 해당 개념을 잠재 공간(latent space)에서 제거하는 방법을 연구합니다.

### 구현 기능

- [ ] SAFREE Baseline 재현
- [ ] UnSafeBench 평가 지표 재현
- [ ] 새로운 Projection 기반 Unlearning 방법 제안
- [ ] concept removal 효과와 image quality preservation 사이의 trade-off 분석

### 어려웠던 점 / 배운 점

- diffusion에서 unlearning을 하는 방법
- training-free 기법에서 PCA 원리를 이해하고 적용하는 과정
- latent space가 실제로 어떤 의미를 갖는지 실험을 통해 파악한 것

## 📫 Contact

- **Email:** rudwns55@g.skku.edu
- **GitHub:** [github.com/kyungjunoh](https://github.com/kyungjunoh)
- **Repository:** [github.com/kyungjunoh/SAFESEXUAL](https://github.com/kyungjunoh/SAFESEXUAL)

---

© 2026 SAFESEXUAL Team · Powered by GitHub Pages
