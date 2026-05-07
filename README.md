# FaceRecall · 民间寻人 AI 画像工具

> AI-guided facial reconstruction tool for civilian missing person cases

用 AI 引导目击者还原一张面孔。适用于家庭寻亲、民间走失协寻、失忆辅助记忆等场景。

---

## 它能做什么

1. **AI 结构化引导问答** — 不需要目击者懂画像，AI 会一步步追问关键特征，自动整理成面部描述
2. **自动生成候选画像** — 问答完成后，图像模型生成 2 张候选画像供对比
3. **多轮迭代精修** — 目击者给反馈，AI 把语言描述翻译成精准的编辑指令，反复迭代直到"像"
4. **导出存档** — 确认收敛后下载最终画像，或导出完整会话记录

```
目击者描述 → AI 引导问答 → 生成初版画像
                                    ↓
              ←── 目击者反馈 ←── 目击者确认
                    ↓
              AI 翻译反馈 → 迭代修图 → 收敛
```

---

## 快速开始

**环境要求**：Python 3.10+，需要能访问 OpenAI 兼容图像 API 的 Key

```bash
git clone https://github.com/jidouqie/FaceRecall.git
cd FaceRecall
./run.sh          # macOS / Linux
# run.bat         # Windows
```

浏览器会自动打开 `http://localhost:8787`。

首次运行会自动创建虚拟环境并安装依赖，之后直接 `./run.sh` 即可。

**配置 API**：点右上角"⚙️ 设置"，填入你的 API Base URL 和 API Key。支持 OpenAI 官方接口及任何兼容接口（如中转网关）。

---

## 界面说明

| 区域 | 功能 |
|------|------|
| 左侧对话区 | AI 引导问答，目击者在此回答 |
| 右侧画像区 | 每轮生成的候选画像，可点击放大 |
| 下方反馈栏 | 选定基准图后提交修改反馈 |

**特征填充率进度条**：显示当前已收集的面部特征完整度，达到足够程度后自动进入生图阶段。

---

## 常见用途

- 家属寻找走失老人、儿童
- 民间寻亲志愿者协助还原面孔
- 目击者辅助记忆（事后还原印象）

---

## 路线图

- [x] AI 结构化引导问答（Guider）
- [x] 多轮图生图迭代精修
- [x] 多模型并行候选图对比
- [x] 线稿 → 真人画像转换
- [x] 历史会话管理
- [x] 会话导出（画像 + JSON）
- [ ] 视觉参考图库（AnchorBank）——辅助目击者选择面部特征
- [ ] PDF 格式寻人启事模板
- [ ] 国产模型支持（豆包、通义）

---

## 免责声明

- 本工具**不构成司法证据**，不替代专业法医画像师
- 请遵守当地法律法规，尊重个人隐私
- 仅限民间寻人、记忆辅助等合法用途
- 所有数据本地存储，不上传任何内容

---

## 开源协议

Apache 2.0 — 可自由使用、修改、分发。

---

## English Summary

FaceRecall is a local web app that uses LLM-guided questioning and image generation to help witnesses reconstruct a face from memory. It runs entirely on your machine — no data leaves your device.

**How it works**: The AI asks structured questions about facial features → builds a description → generates 2 candidate portraits → iterates based on witness feedback until converged.

**Requirements**: Python 3.10+, an OpenAI-compatible API key with image generation access.
