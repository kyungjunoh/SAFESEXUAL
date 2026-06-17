# SAFESEXUAL 🔒

> Training-free nudity removal for text-to-image diffusion models

## 배포 링크

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-배포됨-58a6ff?style=flat&logo=github)](https://kyungjunoh.github.io/SAFESEXUAL/index.html)

## 🧑‍🤝‍🧑 About Us

저희는 성적(sexual)으로 유해한 콘텐츠를 생성하는 확산 모델의 위험성을 줄이기 위한 연구를 수행합니다.
모델 재학습 없이(training-free) 추론 단계에서 누드 개념의 생성을 억제하는 방법을 탐구합니다.

| 이름 | 역할 |
| --- | --- |
| 오경준 | 프로젝트 총괄 · 방법 구현 및 실험 · GitHub Pages 배포 |
| 최정민 | 기획 및 선행 연구 조사 · 벤치마크 분석 |
| 김윤수 | 실험 결과 정리 및 시각화 · 발표 자료 제작 |

`PyTorch` · `Diffusion Models` · `Training-free Safety` · `Cross-Attention` · `CLIP`

## 🚀 Method — IFGR (Image-Feedback Gated Regeneration)

텍스트 기반 안전화 방법은 **적대적 프롬프트**(P4D, MMA-Diffusion 등)에서 한계가 있습니다.
누드 유도 신호가 텍스트에 명시적으로 드러나지 않아, 프롬프트만 봐서는 *언제* 개입할지 알 수 없기
때문입니다. **IFGR**은 개입 여부를 텍스트가 아니라 **실제 생성된 이미지**에 기반해 결정합니다.

1. **Pass 1** — SAFREE로 이미지를 1차 생성 (word-based `P_c`, 데이터 미사용 → leakage 불가)
2. **Gate** — 평가용 NudeNet과 **독립적인** CLIP zero-shot NSFW 검출기로 결과를 스크리닝
3. **Pass 2** — unsafe로 판정된 이미지만 같은 seed에서 **cross-attention 강하게 억제**하여 재생성

benign 프롬프트는 게이트를 통과해 그대로 반환되므로 **생성 품질이 보존**되고, 강한 억제는
누드 이미지에만 적용됩니다. 게이트와 평가 검출기를 분리해 **circular evaluation을 방지**합니다.

### 결과

| | P4D | MMA-Diffusion | 화질 (COCO, FID) |
| --- | --- | --- | --- |
| SD (no defense) | 0.697 | 0.957 | – |
| SAFREE | 0.211 | 0.586 | 6.85 |
| **IFGR (ours)** | **0.183** | **0.542** | **6.85** |

P4D −13%, MMA −7.5%의 상대적 ASR(누드 검출률) 감소를 달성하면서, COCO 9966개 프롬프트에서
SAFREE와 동일한 FID를 유지합니다 (안전성 개선 + 화질 무손상).

## 📂 Structure

```
SAFREE/
├── generate_safree.py              # 메인 파이프라인 (IFGR 포함)
├── models/
│   ├── modified_stable_diffusion_pipeline.py
│   ├── nudity_attn.py              # cross-attention suppression (Pass 2)
│   └── clip_nsfw_gate.py           # 독립 CLIP NSFW 게이트
├── eval_quality.py                 # FID / CLIP score 평가
├── scripts/run_ifgr.sh             # 메인 방법 실행
├── datasets/                       # 평가 벤치마크 (P4D, MMA, Ring-A-Bell, I2P, COCO)
└── PAPER_DRAFT.md                  # 논문 초안
```
> 생성 이미지(`results/`)와 가중치(`pretrained/`)는 용량 문제로 push되지 않습니다 (`.gitignore`).

## ⚙️ Usage

```bash
cd SAFREE
conda activate SAFREE
bash scripts/run_ifgr.sh            # P4D에서 IFGR 실행 → results/.../detect_dict.json (ASR)
```

NudeNet 가중치(`pretrained/nudenet_classifier_model.onnx`)와 SD v1-4가 필요합니다.

### 진행 상황

- [x] SAFREE Baseline 재현
- [x] 누드 벤치마크 평가 (P4D / MMA-Diffusion / Ring-A-Bell / I2P)
- [x] 이미지 피드백 기반 gated regeneration (IFGR) 제안
- [x] concept removal과 image quality preservation 사이의 trade-off 분석

## 깃헙 관련 

### 구현 기능
- 커밋 최소 3회 이상 
- 브랜치 1개 생성
- 소개(About) 작성
- 프로젝트(Project) 내용 추가
- 연락(Contact) 작성
- 내비게이션(상단 메뉴) 생성
- 사이트 소개 작성
- 배포 링크 생성

### 어려웠던 점 / 배운 점
- 체계적인 깃헙 사용 및 활용 방법
- 개인 단위의 수정이 아닌 팀 단위의 수정이 올바르게 이루어지는 방법
- 다양한 방식의 깃헙 접속, 수정, 커밋 방법

## 📫 Contact

- **Email:** rudwns55@g.skku.edu
- **GitHub:** [github.com/kyungjunoh](https://github.com/kyungjunoh)

- **Email:** jeongmin.c5@g.skku.edu
- **GitHub:** [github.com/paparu7](https://github.com/paparu7)

- **Email:** arthur737503@gmail.com
- **GitHub:** [github.com/floatingarthur](https://github.com/floatingarthur)
  
- **Repository:** [github.com/kyungjunoh/SAFESEXUAL](https://github.com/kyungjunoh/SAFESEXUAL)

---

© 2026 SAFESEXUAL Team · Powered by GitHub Pages
