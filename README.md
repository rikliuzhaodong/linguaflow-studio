# FluentFrame

本地视频语言学习工作台。导入中文或英文视频后，会自动开始识别，并将原文、双向翻译和词级时间轴流式加入页面。

## 已实现

- 英文、中文、中英双语、挖空四种学习模式
- 播放时逐句自动跟随，点击字幕可跳转
- 单句循环、上/下一句、播放速度控制
- 单词级时间戳；点击任意单词可循环切换无色、绿色、橙色和蓝色高亮
- 本地视频预览与拖放导入
- Qwen3-ASR-0.6B 中文/英文识别，配合 Qwen3-ForcedAligner-0.6B 生成词级时间戳
- Qwen2.5-3B MLX 4-bit 本地双向翻译（Apple Silicon）
- 每 20 秒音频块完成后立即返回字幕，不必等待整段视频处理完成
- 可选 OpenAI API 翻译；API Key 只保存在本地服务端
- 未导入视频时仍可使用演示课程体验界面

## 环境要求

项目主要在 Apple Silicon Mac 上测试。ASR 也可以在 Linux/macOS 的 CPU 上运行，但本地 MLX 翻译只支持 Apple Silicon；其他平台需要配置 OpenAI API 才能生成双语翻译。

- Node.js `>= 22.13.0`（包含 `npm`）
- pnpm `11.19.0`
- Python `3.11`
- FFmpeg（命令行中需要同时存在 `ffmpeg` 和 `ffprobe`）
- 模型文件约 5 GB，Python/Node 依赖还会占用额外空间

macOS 可以先用 Homebrew 安装基础环境：

```bash
brew install node python@3.11 ffmpeg
```

确认版本：

```bash
node --version
python3.11 --version
ffmpeg -version
```

## 从全新环境安装

以下命令都在仓库根目录执行。

### 1. 安装 pnpm 和前端依赖

全新机器通常没有 `pnpm`，先通过 Node.js 自带的 `npm` 安装：

```bash
npm install --global pnpm@11.19.0
pnpm --version
pnpm install --frozen-lockfile
```

如果没有全局 npm 写入权限，也可以使用 Corepack：

```bash
corepack enable
corepack prepare pnpm@11.19.0 --activate
pnpm install --frozen-lockfile
```

### 2. 创建 ASR Python 环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

以后不需要手动激活这个环境；`pnpm dev` 会使用现有的 `/Users/rik/workspace/interview_summary/.venv/bin/python`，也可以通过 `FLUENTFRAME_ASR_PYTHON` 覆盖。

### 3. 下载本地模型

模型权重单个超过 GitHub 的普通文件大小限制，因此不会提交到代码仓库。安装 Python 依赖后，使用 Hugging Face CLI 下载：

```bash
mkdir -p models

hf download Qwen/Qwen3-ASR-0.6B \
  --local-dir models/Qwen3-ASR-0.6B

hf download Qwen/Qwen3-ForcedAligner-0.6B \
  --local-dir models/Qwen3-ForcedAligner-0.6B
```

如果 `hf` 下载速度不稳定，可设置 Hugging Face 镜像或手动从模型页面下载，但目录名必须保持不变。

模型页面：[Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)、[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)。

### 4. 配置翻译

二选一即可。

#### 方案 A：Apple Silicon 本地翻译

MLX 和 ASR 所需的 Transformers 主版本不同，因此本地翻译必须使用第二个虚拟环境：

```bash
python3.11 -m venv local-asr/.venv-mlx
local-asr/.venv-mlx/bin/python -m pip install --upgrade pip
local-asr/.venv-mlx/bin/python -m pip install -r local-asr/requirements-mlx.txt

.venv/bin/hf download mlx-community/Qwen2.5-3B-Instruct-4bit \
  --local-dir models/Qwen2.5-3B-Instruct-4bit
```

模型页面：[mlx-community/Qwen2.5-3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit)。

#### 方案 B：OpenAI API 翻译

不需要下载 Qwen2.5 翻译模型，也不需要创建 MLX 虚拟环境：

```bash
cp .env.example .env.local
```

然后编辑 `.env.local`：

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_TEXT_MODEL=gpt-5-mini
```

### 5. 启动

```bash
pnpm dev
```

打开 <http://localhost:3000>，选择视频后识别会自动开始。

`pnpm dev` 会同时启动网页和本地 ASR 服务。首次识别会加载数 GB 模型，等待时间会明显长于后续请求。

启动后请等待终端出现 `Local: http://127.0.0.1:3000/` 再打开浏览器。如果提示 `Another vinext dev server is already running`，先在旧服务终端按 `Ctrl+C`，不要同时运行两个 `pnpm dev`。

## 单独运行和检查 ASR

只启动 ASR 服务：

```bash
pnpm asr
```

检查服务：

```bash
curl http://127.0.0.1:8766/health
```

不要在浏览器打开 `http://127.0.0.1:8766/`：这是没有网页的 ASR API 根路径，返回 `{"detail":"Not Found"}` 属于正常现象。网页地址是 <http://localhost:3000>；`/health` 和 `/transcribe` 才是 ASR 接口。

## 可选配置

服务设备策略默认为 `auto`：检测到 Apple MPS 时使用 GPU，否则回退 CPU。可以在运行命令前通过环境变量覆盖，例如：

```bash
FLUENTFRAME_QWEN_DEVICE=cpu pnpm dev
```

其他可用环境变量：

- `FLUENTFRAME_QWEN_ASR_MODEL`：ASR 模型目录
- `FLUENTFRAME_ALIGNER_MODEL`：ForcedAligner 模型目录
- `FLUENTFRAME_ASR_PYTHON`：ASR Python 路径
- `FLUENTFRAME_TRANSLATOR_PYTHON`：MLX 翻译 Python 路径
- `FLUENTFRAME_FFMPEG` / `FLUENTFRAME_FFPROBE`：FFmpeg 工具路径
- `FLUENTFRAME_MAX_UPLOAD_BYTES`：最大上传字节数，默认 2 GB
- `FLUENTFRAME_CHUNK_SECONDS`：音频分块秒数，默认 20

示例：

```bash
FLUENTFRAME_ASR_PYTHON=/path/to/python \
FLUENTFRAME_FFMPEG=/path/to/ffmpeg \
pnpm dev
```

## 处理流程

本地服务先用 FFmpeg 将视频按 20 秒切成音频块，再通过 Qwen3-ASR 生成中英文原文；ForcedAligner 提供词级时间戳，字幕按停顿合并成句。每完成一个音频块，结果会立即推送到浏览器。

Apple Silicon 可使用本地 MLX Qwen2.5-3B 完成双向翻译；配置 OpenAI API 后，服务会改为按音频块调用云端翻译。
