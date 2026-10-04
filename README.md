# subalign: 精准逐字字幕 / 歌词对齐工具

把音频或视频变成**逐字（字符级）对齐**的字幕和歌词。可以自动识别语音，也可以用你提供的稿件或歌词。

- **两种输入**：自动语音识别（Whisper / faster-whisper / FunASR Paraformer / OpenAI 兼容 API），或者用户稿件、歌词（txt / lrc / srt / qrc / krc …）
- **自动校对**：拿稿件和实际读、唱出来的内容做比对，生成差异报告（漏读、多读、改词、同音误识）；没有稿件时用大模型修正识别错误，修正后时间轴不变
- **歌曲对齐**：先分离人声，再结合音高变化和起音检测，用全局动态规划做**逐字 + 逐行**对齐（详见下文“算法”）
- **横屏 / 竖屏**：按分辨率和字号算出每行最大宽度，用全局最优断句保证不超出屏幕；竖屏单独断句
- **多种导出格式**：SRT / VTT / ASS / LRC / 增强 LRC / 逐字 LRC / QRC / KRC（含加密 .krc）/ YRC / TTML（Apple 逐字）/ SBV / JSON / TXT
- **特效字幕样式**：ASS 卡拉 OK（`\k` `\kf` `\ko`），逐字弹跳、辉光、打字机等特效；内置预设，也可以用 JSON 自定义；已有的特效字幕可以读进来再换样式
- **大模型翻译**：Claude（官方 SDK），以及 OpenAI / DeepSeek / 通义千问 / Kimi / 智谱 / Ollama 等 OpenAI 兼容接口。按行翻译，不打乱时间轴，可以导出双语字幕
- **人声分离**：BS-RoFormer（audio-separator）/ Demucs / 无依赖 DSP 兜底，可以单独导出人声或伴奏

## 安装

```bash
pip install -e .                   # 核心（numpy / scipy / soundfile），需要系统里有 ffmpeg
pip install -e ".[asr,zh]"         # + faster-whisper 识别、拼音/分词（推荐）
pip install -e ".[ctc]"            # + CTC 强制对齐（torch + transformers），精度最高
pip install -e ".[separate]"       # + Demucs 人声分离（或 pip install audio-separator 用 RoFormer）
pip install -e ".[llm]"            # + Claude SDK（翻译 / 校对）
pip install -e ".[all]"
```

所有重型依赖都是可选的，缺少的模块会自动降级（见“精度来源”）。

## 快速开始

```bash
# 1. 口播视频：自动识别 + 逐字对齐，同时输出横屏和竖屏两套字幕
subalign align talk.mp4 --lang zh --layout landscape,portrait

# 2. 有稿件：按稿件对齐，并输出校对报告 talk.proofread.md
subalign align talk.mp4 --script script.txt --punct space

# 3. 歌曲 + 歌词：逐字和逐行对齐，导出常见歌词格式
subalign lyrics song.mp3 --lyrics lyrics.txt --title 晴天 --artist 周杰伦 \
    -f lrc,lrc-enhanced,qrc,krc-encrypted,yrc,ttml,ass --style karaoke-pop

# 4. 歌曲没有歌词：识别歌词，再翻译成英文，输出双语
subalign lyrics song.mp3 --translate en --llm-provider deepseek

# 5. 人声分离：输出人声和伴奏
subalign separate song.mp3 --stems vocals,instrumental --format flac

# 6. 已有字幕：换样式、改成竖屏、转格式
subalign convert old.ass -f ass,srt --style neon --layout 1080x1920

# 7. 翻译 / 校对已有字幕
subalign translate movie.srt --to en --llm-provider anthropic
subalign proofread asr.srt --context "AI 技术分享，嘉宾：张三" --glossary terms.json

subalign formats   # 列出全部格式、样式预设和画面布局
```

## 算法

### 1. 多来源锚点

“锚点”是每个字的大致起始时间，按精度从高到低依次尝试：

| 来源 | 典型误差 | 条件 |
|---|---|---|
| CTC 强制对齐（wav2vec2 / MMS） | ±40 ms | `torch` + `transformers` |
| ASR 词时间戳和稿件做文本对齐 | ±150 – 300 ms | 任一 ASR 后端 |
| 稿件自带的行时间（LRC / SRT） | 行级 | 输入带时间 |
| 无（纯声学） | — | 只用于歌曲 |

- **CTC Viterbi**（`align/ctc.py`）：用 numpy 实现。在行和行之间插入可跳过的 `<star>` 垃圾状态，用来吸收和声、即兴、口白这类稿件里没有的声音。模型词表里没有的字会自动转写：中文转拼音，日文转罗马字，韩文按谚文字母转写，其他语言去掉变音符号。所以只有拉丁字母词表的 MMS 模型也能对齐中文、日文、韩文。
- **文本对齐**（`align/sequence.py`）：先用 difflib 找出完全一致的片段，片段之间的空隙再用带权编辑距离 DP 对齐。替换代价是 `1 − 拼音相似度`，并且支持 zh/z、n/l、in/ing 这类模糊音，所以同音误识（岗/刚）能正确对上，在校对报告里标为低严重度。

### 2. 逐字 DP 精修（核心，`align/syllable_dp.py`）

歌词逐字对齐难在三点：一个字可能拖好几秒；连唱（legato）换字时没有能量起音；行与行之间夹着间奏。处理方法如下。

**候选边界**取下面几类的并集：

- 频谱通量起音（SuperFlux：先在频率方向做最大值滤波，抑制颤音引起的假起音）
- 音高换音点（YIN 音高轨迹做中值滤波后的半音跳变，用来捕捉连唱换字）
- 人声活动的上升沿
- 有声区内的密集网格，加上静音区内的稀疏网格（保证总有可行解）

**半马尔可夫 DP**为每个字选一个候选点，最小化：

```
Σ unary(i, c_i)               起音强度奖励 / 在静音中起字的惩罚 / 行首在停顿之后的奖励
                              + 和锚点距离的 Huber 代价（按锚点置信度和误差 σ 加权）
+ Σ trans(i, c_{i-1}, c_i)    上一个字的“有声时长”对数正态似然（行尾字用拖长音的统计参数，
                              间奏的静音不计入时长）
                              + 字内被跳过的强起音（说明可能漏切了边界）
                              + 行中字内出现静音（换气允许，但有代价）
```

DP 在锚点附近做带状限制。没有锚点时，按“有声时长”把字的累计权重成比例映射成先验，再在先验附近做带状限制。复杂度是 O(N·W·B) 次向量运算，一首 4 分钟的歌约 1 秒。字的结束时间取下一个字的开始时间，如果中间遇到静音（行尾、换气），就截断在人声消失的位置。

在合成的“歌唱”测试集上（含颤音、连唱、拖长音、间奏），**纯声学、不用任何锚点**时 100% 的字起始误差在 50 ms 以内，平均误差约 8 ms；锚点误差 σ=300 ms 时，仍有 94% 在 50 ms 以内。

### 3. 横竖屏断句（`segment/linebreak.py`）

采用 Knuth-Plass 式的全局最优断行：

- 每行宽度必须 ≤ 版面宽度。CJK 字符计 2 个单位，拉丁字母计 1 个；`单位数 = 分辨率宽 × 安全区比例 ÷ (字号 / 2)`
- 倾向在句末标点、逗号、语音停顿处断开
- 尽量不切开中文词语（安装 jieba 时生效），从不切开英文单词，避免孤字、上下行长短悬殊
- 照顾时长和阅读速度：避免过短、过长的条目，限制每秒字数（CPS）

竖屏默认 1080×1920、每行约 14 个汉字、每条最多 2 行，位置抬高，避开平台界面。也可以直接指定分辨率，例如 `--layout 720x1280`，或者 `--max-chars`、`--font-size`、`--max-lines`。歌曲一行切成多段后，每段各自成为一行，并保留逐字时间。

## 格式

| 格式 | 级别 | 说明 |
|---|---|---|
| `srt` / `vtt` / `sbv` / `txt` | 行 | 通用字幕，支持双语 |
| `srt-karaoke` | 字 | 每个字一条，用 `<font>` 高亮已唱部分 |
| `vtt-karaoke` | 字 | WebVTT 行内时间戳 `<00:00:01.000>`，配 `::cue(:past)` 样式 |
| `ass` | 字 | 样式 + 卡拉 OK + 特效，双语 |
| `lrc` | 行 | 标准 LRC，双语用相同时间戳的两行，支持读取压缩写法 `[t1][t2]` 和 `offset` |
| `lrc-enhanced` | 字 | A2 增强 LRC `<mm:ss.xx>`（ESLyric / foobar 等） |
| `lrc-word` | 字 | 逐字 LRC `[mm:ss.xx]字[mm:ss.xx]字` |
| `qrc` | 字 | QQ 音乐（明文；可加 XML 外壳） |
| `krc` / `krc-encrypted` | 字 | 酷狗（明文 / 加密 .krc），翻译写入 `[language:]` |
| `yrc` | 字 | 网易云逐字歌词 |
| `ttml` / `ttml-line` | 字 / 行 | Apple Music 风格 TTML，翻译写为 `x-translation` |
| `json` | 字 | 完整数据，可以再导入 |

除了 `srt-karaoke`，以上格式都能读入，所以本工具也可以当作格式转换器使用。

## 样式

预设：`default`、`box`、`karaoke`、`karaoke-pop`、`neon`、`typewriter`、`bounce`、`shortvideo`

```json
{
  "preset": "karaoke",
  "main":        {"fontname": "思源黑体 Heavy", "primary": "#FFD34D", "secondary": "#FFFFFF",
                  "outline_color": "#202020", "outline": 4, "bold": true},
  "translation": {"fontsize": 40, "primary": "#E0E0E0"},
  "translation_position": "below",
  "effect": {"karaoke": "kf", "syllable": "pop", "pop_scale": 120,
             "lead_in_ms": 600, "lead_out_ms": 300, "fade_in_ms": 150, "fade_out_ms": 150}
}
```

`effect.karaoke` 可选：`none` | `k` | `kf` | `ko`

`effect.syllable` 是逐字特效，可选：`none` | `pop` | `bounce` | `glow` | `typewriter`

颜色写成 `#RRGGBB` 或 `#RRGGBBAA`（AA 表示不透明度）。字号和边距会按横屏或竖屏的分辨率自动缩放。

## 大模型

| `--llm-provider` | 接口 | 环境变量 |
|---|---|---|
| `anthropic`（默认） | Claude 官方 Python SDK，默认模型 `claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| `openai` / `deepseek` / `qwen` / `moonshot` / `zhipu` / `ollama` | OpenAI 兼容 Chat Completions | `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / `MOONSHOT_API_KEY` / `ZHIPUAI_API_KEY` |
| `custom` | 任意兼容接口：`--llm-base-url` + `--llm-model` | `LLM_API_KEY` |

翻译按批次发送。每批行带编号，并附上前后几行作为上下文；返回结果里缺少的编号会逐行重试，保证译文和原文一行对一行。

## Python API

```python
from subalign.align.aligner import AlignConfig, align_audio
from subalign.formats import save
from subalign.segment.layout import get_layout
from subalign.segment.linebreak import segment_document

res = align_audio("song.mp3", open("lyrics.txt", encoding="utf-8").read(),
                  AlignConfig(mode="song", language="zh"))
doc = segment_document(res.document, get_layout("portrait"))
save(doc, "song.qrc")
save(doc, "song.ass", style=None, layout=get_layout("portrait"))
```

## 精度来源与降级

| 组件 | 最佳 | 降级 |
|---|---|---|
| 锚点 | CTC（`[ctc]`） | ASR 词时间戳 → 稿件时间 → 纯声学 |
| 人声分离 | `audio-separator`（BS-RoFormer）/ Demucs | DSP：立体声中置提取（效果可用）；单声道 REPET-SIM（较弱） |
| 中文 | `pypinyin`（同音感知对齐、CTC 转写）+ `jieba`（不切开词语） | 按字符直接比对 |

模型（Whisper、wav2vec2、Demucs、RoFormer）第一次使用时会从网络下载。

## 开发

```bash
pip install -e ".[dev]"
pytest
```

测试用合成的“歌唱 / 语音”音频（`tests/synth.py`），覆盖以下内容：逐字 DP 精度、CTC Viterbi、全部格式的往返读写、断句、样式、分离，以及 ASR、LLM、CTC 打桩的端到端流程。
