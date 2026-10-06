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

# 8. 语音粗剪：剪掉语气词、口吃重复、重录句和过长停顿（视频同步剪辑），并输出剪后字幕
subalign roughcut talk.mp4 --lang zh --level standard -f srt,ass
subalign roughcut talk.mp4 --plan output/talk.roughcut.json   # 按审阅后的剪辑计划重新渲染

subalign formats   # 列出全部格式、样式预设和画面布局
```

## 高精度逐字稿

```bash
subalign align talk.mp4 --mode speech --lang zh --diarize --cross-check funasr \
    --context "AI 访谈，嘉宾张伟" -f transcript,docx,srt,json
```

- **两遍得到逐字时间**：先由识别模型写出文字，再用 CTC 把这些文字强制对齐到音频，得到每个字的起点（约 ±40 ms）；对齐结果和识别模型自己给出的时间核对，差距过大就退回识别时间。只用识别时间时，停顿会被放到错误的位置。
- **识别提示**：`--context` 和术语表里的人名、术语会作为提示交给识别模型；`--verbatim`（严格逐字）让 Whisper 保留嗯/呃、重复和半句重说，大模型校对时也会保留这些内容。
- **幻觉过滤**：对每一段识别结果综合打分，依据有：字幕署名 / 频道宣传语、该时段有没有人声、循环重复、语速异常、识别置信度。分数够高的直接删除，处于边缘的标为待核对，都写进 `.asr-report.md`。实测能删掉「优优独播剧场——YoYo Television Series Exclusive」这类凭空出现的文字。`--keep-hallucinations` 关闭自动删除。
- **说话人区分**：`--diarize`（或 `--speakers N`）用 CAM++ 声纹（3D-Speaker，经 ModelScope 下载，无需申请权限）加聚类；一行里如果换了说话人，会按停顿把这一行拆开。
- **双模型交叉核对**：`--cross-check funasr` 用第二个识别模型再识别一遍。两者不一致的字会被标为低置信度（网页里用橙色标出，Word 里黄色高亮），并列在质检报告和逐字稿末尾，人工只需检查这些地方。
- **逐字稿格式**：`transcript`（Markdown）和 `docx`（Word），按说话人分段，每段带时间戳。

## 口播音频：录音棚 + 音频处理

**录音棚**（网页）：
- 选择录音设备、采样率（44.1 / 48 kHz）、单声道或立体声、保存位深；浏览器的回声消除、降噪、自动增益都已关闭，录到的是原始信号，界面上有输入电平表和削波提示。
- 空格键录制 / 停止，停止即暂停，可以先回放（回车），再接着录。
- 在波形上点击可以放光标，拖动可以选区；选中一段后按空格，会先播放预卷，再录音替换这一段（新录的长度可以和原来不同），两处接缝都做等功率交叉淡化。
- 支持删除选区和撤销 / 重做，录完可以下载 WAV，或直接送去「音频处理」。

**音频处理**（`subalign studio`，网页「音频处理」页面），按后期常规顺序处理：

| 步骤 | 做法 |
|---|---|
| 低切 | 80 Hz 高通，24 dB/oct |
| 降噪 | RNNoise 语音模型（`models/rnnoise`）/ 频谱降噪 |
| 智能清理 | 气口削弱 12 dB 或删除、咳嗽删除、口头禅删除（需要语音识别）、长停顿缩短；在降噪之后检测，更准 |
| 去咔哒 | 检测孤立冲击（比前后 ±10 ms 内最响处还高 3 倍），用中值滤波修复；人声的声门脉冲是周期性的，不会被误判 |
| 去喷麦 | 动态低频压缩：40–160 Hz 比该说话人平时的低频高 12 dB 以上、且低频压过中频时才压，低沉的嗓音不会被削薄 |
| 去齿音 | 动态 STFT 去齿音：4.5–10 kHz 的频段比 200–4500 Hz 的人声主体还响时，最多衰减 6 dB（可调） |
| EQ | 250 Hz 处 -3 dB 去闷，3 kHz 处 +2.5 dB 提升清晰度，可选 10 kHz 空气感 |
| 压缩 | 3:1，阈值 -20 dBFS |
| 混响 | 可选，合成小房间的冲激响应，干湿比可调 |
| 背景音乐 | 循环或截断到人声长度，淡入淡出；响度设为人声的 15–25%（默认 20%，约低 14 dB），人声说话时再自动压低 |
| 响度标准化 | EBU R128 积分响度（-14 / -16 / -18 / -23 LUFS），加 4 倍过采样的真峰值限幅（默认 -1.5 dBTP） |
| 导出 | WAV（16 / 24 / 32 bit）、MP3、AAC（m4a）、FLAC、Opus；采样率和声道可选 |

每个步骤都会把实测结果（底噪、修复处数、响度、真峰值）写进 `.studio.json`；网页上有 A/B 对比试听，可以按响度对齐后再比较。音频处理优先使用项目内的稳定版 ffmpeg（`tools/ffmpeg`），因为开发版 ffmpeg 的 RNNoise 滤镜会偶发崩溃或输出 NaN。

```bash
subalign studio take.wav --cleanup --bgm music.mp3 --loudness -16 --format mp3 --bitrate 320k
```

## AI 配音（网页「AI 配音」页面）

使用的引擎是 [IndexTTS-2.5](https://github.com/index-tts/index-tts)。它用参考音频 A 克隆音色，再用参考音频 B 复刻情绪，两者可以分开指定；另外支持 `<字|拼音>` 注音和语速控制。IndexTTS 需要 torch 2.8，所以装在独立环境 `tools/index-tts/.venv` 里，由常驻子进程 `webui/tts_worker.py` 调用，模型只加载一次。权重放在 `models/indextts-2.5`，`setup.bat` 会自动安装。

1. **文稿预处理**（一键完成）：
   - 删除括号备注、【提示】、表情、链接和 Markdown；
   - 数字转成读法：1w+ → 一万多，3.5% → 百分之三点五，2026-10-06 → 二零二六年十月六日，10:30 → 十点三十分，¥99 → 九十九元，3-5个 → 三到五个，1/3 → 三分之一；手机号、座机号逐位读；
   - 符号转成文字，标点统一为全角；
   - 分句：长句按逗号拆开，过短的句子并入前一句（自回归 TTS 生成一两个字时不稳定）；
   - 列出常见多音字，并给出按上下文推荐的读音。
2. **手动标记**：多音字 `<行|hang2>`，停顿 `<停|0.5>`，重读 `<重|关键词>`。
3. **逐句生成**：
   - 每句单独生成，语速可调范围 0.95–1.05×；
   - 生成后自动校验：用语音识别听写这一句，再和原文比对（同音字不算错），误读超过 12% 就换随机种子重新生成，最多 3 次，保留误读最少的一版；
   - 每句都能试听、修改文字或停顿后单独重新生成，也能切换回历史版本；
   - 改动后总音频会自动重新合成。
4. **去机械感**，在合成总音频时处理：

| 问题 | 处理 |
|---|---|
| 句首句尾有杂音、残留呼吸 | 按能量修剪每一句，边缘加淡入淡出 |
| 各句音量不一 | 逐句做响度匹配，调整幅度不超过 ±6 dB |
| 各句语速忽快忽慢 | 和中位语速相差超过 6% 的句子做保持音高的时间伸缩 |
| 停顿过于整齐 | 按标点安排停顿，再加 ±10% 的随机变化 |
| 没有换气声 | 从音色参考 A 里提取说话人自己的真实换气声，放在较长停顿后、下一句之前 |
| 句间是纯数字静音 | 铺一层 -66 dBFS 的粉红房间底噪 |
| 声码器的金属感 | 7.5 kHz 以上做高架衰减；再用激励器补回 22 kHz 模型缺失的 11 kHz 以上泛音（空气感） |
| 声音干、薄 | 轻度谐波饱和（温暖度） |
| 重读不明显 | 用 CTC 定位 `<重|…>` 标记的词，提升约 3 dB |

5. **后期处理**：点「送去音频处理」后，沿用口播的处理链（低切、EQ、压缩、响度、背景音乐、导出），并自动选用「AI 配音」预设。这个预设关闭降噪、清理和去咔哒，因为 AI 配音没有这些问题；去齿音和清晰度提升也调得更轻。未处理的总音频可以直接导出为 WAV、MP3、AAC、FLAC 或 Opus。

## 音频翻译 / 配音翻译（网页「音频翻译」页面）

录音 / 视频 → 语音识别（「语音对齐」流程，逐字时间）→ 翻译 → 原说话人音色 AI 配音，对齐原视频时间轴。
也可以直接翻译字幕（保留时间轴）或文稿；在「AI 配音」页点「翻译后配音」把当前文稿带过来。中英互译都适用。

翻译要同时满足两个约束（`subalign/translate/isochrony.py`）：

* **时长（音节数）**：数出原句的音节 / 音素（中文按拼音，英文按元音组规则；数字按读法、缩写逐字母），
  目标音节数 = 原句音节 × 两种语言平均语速之比（中 5.18、英 6.19 音节/秒 …）。原句有时间轴且说得快时，
  再按「时长 × 配音的正常语速」封顶，避免配音超出原句。允许偏差默认 ±10%。
  按音节而不是音素定长：英文一个音节的音素比中文多，但说得一样快，时长跟着音节走。
* **名词位置（语序重写）**：大模型标出名词 / 人名 / 数字（锚点，jieba 兜底），按逐字时间算出每个锚点在原句里
  出现的位置（例：10 秒的句子第 3 秒 → 30%），要求译文把它放在同一半句、相近位置，必要时改写语法
  （前置、被动、同位语、拆成两个短句）。

每句让大模型写多个结构不同的候选 → 逐个计数打分（音节偏差、锚点位置、是否跨半句）→ 未达标的句子带着具体问题
（"多了 6 个音节"、"Apple 在 85%，应在前半句"）再修正 → 大模型从最优的几个里挑最自然的。
可以逐句选其他候选、手改（自动重新计数）、写提示再生成；导出译文 / 双语 SRT。

配音：每句一个片段；「对齐原视频时间轴」时每句从原句开始处说，超出时变速（最多 ×1.15，保持音高），
仍放不下才顺延。用本地 Ollama 翻译时会先释放识别模型和空闲的 TTS 进程，开始配音前卸载大模型（11 GB 显卡放不下全部）。

## 语音粗剪

`subalign roughcut` 识别并剪掉口播中的冗余部分，同时保证剪完听起来连贯：

- **识别**：Whisper 加一段带口语词的提示，让它把 嗯 / 呃 等写进识别结果；然后用 CTC 对识别出的文字做强制对齐，得到每个字的起点；每个字的终点和停顿长度按能量包络实测，不靠推算。
- **删什么**：
  - 语气词：嗯 / 呃 / 额 / um 一律删除；
  - 啊、哦：只删前面有停顿的，「好啊」这类句尾语气词保留；
  - 那个、就是说：只删后面跟停顿的，「那个人」保留；
  - 不拆词：「额度」「额外」这类词不会被当成语气词（jieba 词典判断）。
  - 重复和口吃（「我我我觉得」「我们今天，我们今天要讲」），只保留最后一遍；「谢谢」「看看」这类叠词不算口吃。
  - 重录句：同一句话马上又说一遍，只保留最后一遍。
  - 识别器漏掉的发声，多数是没被写下来的 嗯 / 呃。
  - 过长停顿：缩短到自然长度。
  - 可选：用大模型标记多余的句子（`--llm`）。
- **剪得顺**：
  - 剪点落在附近最安静的时刻；
  - 两个词之间保留自然停顿，长度取自原来的停顿；
  - 被删内容的两侧原本没有停顿时，补一小段这份录音自己的环境底噪，不直接把两个词硬接在一起；
  - 每个剪点都做等功率交叉淡化：静音处 25 ms，剪在发声中时 40 ms。
- **输出**：
  - 剪后的音频；
  - 视频输入还会输出剪后视频，切点按帧对齐，与音频误差不到 1 帧；
  - 按剪后时间轴重新计时的字幕；
  - 剪辑计划 `.roughcut.json` 和剪辑报告 `.roughcut.md`。
- **人工审阅**：网页里被剪的内容显示为划线。单击可以恢复或删除，双击试听原音频；按类别整体开关，调整停顿参数后点「重新生成」，只重新渲染，不会重新识别。

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

## 功能测试台（Web）

`webui/` 是一个本地网页，每个功能都可以在页面上运行：语音对齐、歌词对齐、人声分离、格式转换、翻译、校对、格式列表，以及 pytest。页面直接调用 `subalign` CLI，并显示等效命令；对齐结果可以边播放音频边看逐字高亮。用“生成合成测试音频”时，页面会对比真值，算出逐字误差。

- **特效字幕预览**：导出的 `.ass` 用 libass（WebAssembly，`webui/static/vendor/octopus`，MIT）渲染，效果和播放器一致；输入是视频时字幕叠加在视频上，中文用系统字体（微软雅黑 / 黑体）兜底。
- **人工校对**：对齐结果下方可以逐行修改文字（增删、拆分、合并行，修改起始时间；也可以切到 LRC 文本模式），然后“按修改后的文本重新对齐”。改好的文本会作为带时间的稿件或歌词重新逐字对齐：原来的行时间用来核对 CTC 结果，也作为兜底；歌曲会直接复用已分离的人声。如果文本是自动识别出来的，这个面板会默认展开，并用橙色标出有低置信度字的行。

```bat
setup.bat    :: 一次性：.venv + 全部依赖（CUDA torch）+ 便携 Ollama + 下载全部模型
start.bat    :: 启动 http://127.0.0.1:7860 ，同时启动项目内的 Ollama
```

所有模型都放在项目的 `models/` 目录（`webui/paths.py` 设置各类缓存路径）：

| 目录 | 内容 |
|---|---|
| `models/huggingface` | faster-whisper large-v3 / large-v3-turbo，wav2vec2 / MMS CTC 模型 |
| `models/cache/whisper` | openai-whisper turbo |
| `models/modelscope` | FunASR Paraformer + VAD + 标点 |
| `models/torch` | Demucs htdemucs |
| `models/audio-separator` | BS-RoFormer |
| `models/ollama` | 本地大模型 qwen2.5:7b（翻译 / 校对，`--llm-provider ollama`） |

单独补下载：`.venv\Scripts\python webui\download_models.py whisper ctc`（可选组：whisper、openai-whisper、ctc、demucs、uvr、funasr、llm）。

## 开发

```bash
pip install -e ".[dev]"
pytest
```

测试用合成的“歌唱 / 语音”音频（`tests/synth.py`），覆盖以下内容：逐字 DP 精度、CTC Viterbi、全部格式的往返读写、断句、样式、分离，以及 ASR、LLM、CTC 打桩的端到端流程。
