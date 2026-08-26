"""Trimmed port of MoneyPrinterTurbo's app/services/voice.py.

v1 scope is edge-tts only (free, no API key, subtitle timing comes straight
from edge-tts's ``SubMaker``). Kept: ``azure_tts_v1`` (despite the legacy
name, this is MPT's actual edge-tts implementation -- it's what every plain
edge-tts voice name falls through to in the upstream ``tts()`` dispatcher),
``create_edge_tts_communicate``, ``parse_voice_name``,
``convert_rate_to_percent``, ``create_subtitle`` and their private helpers,
plus ``get_audio_duration`` (used by the pipeline to size material download).
The ``tts()`` dispatcher itself is trimmed to just the edge-tts branch --
all the other MPT TTS providers (Azure v2, SiliconFlow, Gemini, MiMo,
MiniMax, ElevenLabs, Chatterbox, Fish Audio, "no voice") are out of scope
for v1 and were dropped along with the functions that only exist to serve
them.
"""

import asyncio
import inspect
import math
import os
import queue
import re
import threading
import time
from typing import Union
from xml.sax.saxutils import unescape

import edge_tts
from edge_tts import SubMaker
from loguru import logger
from moviepy.video.tools import subtitles
from moviepy.audio.io.AudioFileClip import AudioFileClip

from . import mpt_config as config
from . import utils

_DEFAULT_EDGE_TTS_TIMEOUT_SECONDS = 30.0


def mktimestamp(time_unit: float) -> str:
    """
    将 edge_tts 使用的 100 纳秒时间单位转换为字幕时间戳。

    edge_tts 7.x 不再导出旧版本里的 `mktimestamp`，但项目里旧字幕链路
    还需要这个格式化函数来兼容手工构造的字幕时间轴，因此这里内置一个等价实现。
    """
    hour = math.floor(time_unit / 10**7 / 3600)
    minute = math.floor((time_unit / 10**7 / 60) % 60)
    seconds = (time_unit / 10**7) % 60
    return f"{hour:02d}:{minute:02d}:{seconds:06.3f}"


def parse_voice_name(name: str):
    # zh-CN-XiaoyiNeural-Female
    # zh-CN-YunxiNeural-Male
    name = name.replace("-Female", "").replace("-Male", "").strip()
    return name


def ensure_file_path_exists(file_path: str) -> None:
    """
    确保输出文件所在目录一定存在。

    这里单独做一层兜底，是因为 edge_tts 7.x 在真正发起网络请求之前，
    就会先打开目标音频文件；如果目录不存在，会直接因为本地文件路径报错，
    从而掩盖真正的 TTS 行为结果。
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def create_edge_tts_communicate(
    text: str, voice_name: str, rate_str: str
) -> edge_tts.Communicate:
    """
    按当前已安装的 edge_tts 版本构造 Communicate 对象。

    背景：
    1. 主线代码已经升级到 edge_tts 7.x，并使用 `boundary` 参数拿到更细的边界事件；
    2. 但如果依赖更新失败，现场环境可能仍然停留在旧版 edge_tts；
    3. 旧版 `Communicate.__init__()` 不接受 `boundary`，会直接抛出
       `unexpected keyword argument 'boundary'`，导致整个 TTS 链路失败。

    因此这里先根据构造函数签名探测当前版本支持的参数，再决定是否传入
    `boundary`，让同一份代码同时兼容旧版和新版依赖。
    """
    communicate_kwargs = {"rate": rate_str}
    communicate_signature = inspect.signature(edge_tts.Communicate)

    if "boundary" in communicate_signature.parameters:
        communicate_kwargs["boundary"] = "WordBoundary"

    return edge_tts.Communicate(text, voice_name, **communicate_kwargs)


def get_edge_tts_timeout_seconds() -> Union[float, None]:
    """
    获取 edge-tts 单次流式请求的超时时间。

    背景：
    Edge consumer TTS 在网络不通、服务端限流、voice 与文本语言不匹配等场景下，
    可能长时间卡在 `stream_sync()` 内部，日志只停留在 `start`。这里提供一个
    默认超时，避免任务长期无反馈。

    使用方式：
    - 默认 30 秒，覆盖常见短视频脚本的首包等待时间；
    - 如用户处于慢网络或代理环境，可在 config.app 里设置 `edge_tts_timeout`；
    - 设置为 0 或负数表示显式禁用超时，保留完全向后兼容。
    """
    raw_timeout = config.app.get(
        "edge_tts_timeout", _DEFAULT_EDGE_TTS_TIMEOUT_SECONDS
    )
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError):
        logger.warning(
            "invalid edge_tts_timeout: "
            f"{raw_timeout}, fallback to {_DEFAULT_EDGE_TTS_TIMEOUT_SECONDS}s"
        )
        timeout_seconds = _DEFAULT_EDGE_TTS_TIMEOUT_SECONDS

    if timeout_seconds <= 0:
        return None

    return timeout_seconds


def _stream_edge_tts_sync_with_timeout(
    communicate, on_chunk, timeout_seconds: float
) -> None:
    """
    带总超时地消费 edge_tts 7.x 的同步流。

    实现原因：
    `stream_sync()` 本身是阻塞迭代器，网络层卡住时主线程无法及时恢复。
    这里把阻塞迭代放到 daemon 线程中，主线程通过 Queue 获取 chunk，
    到达超时时间后直接抛出 TimeoutError，让外层重试和错误日志继续工作。

    注意：
    daemon 线程只作为兜底保护使用，最多随 azure_tts_v1 的 3 次重试产生
    少量残留线程；进程退出时会自动回收。相比任务永久卡住，这是
    更可控的失败模式。
    """
    stream_queue = queue.Queue()
    done_marker = object()

    def _produce_chunks():
        try:
            for chunk in communicate.stream_sync():
                stream_queue.put(("chunk", chunk))
            stream_queue.put(("done", done_marker))
        except Exception as e:
            stream_queue.put(("error", e))

    thread = threading.Thread(target=_produce_chunks, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError(
                f"edge_tts stream timed out after {timeout_seconds:g}s"
            )

        try:
            item_type, payload = stream_queue.get(
                timeout=min(0.5, remaining_seconds)
            )
        except queue.Empty:
            continue

        if item_type == "chunk":
            on_chunk(payload)
        elif item_type == "error":
            raise payload
        elif item_type == "done":
            return


def stream_edge_tts_chunks(
    communicate, on_chunk, timeout_seconds: Union[float, None] = None
) -> None:
    """
    统一消费 edge_tts 的同步流和旧版异步流。

    edge_tts 7.x 提供 `stream_sync()`，可以在同步函数里直接迭代；
    更早的版本通常只有异步 `stream()`。为了让 `azure_tts_v1()` 在
    旧依赖残留场景下仍能继续工作，这里统一做一层流式兼容。

    Args:
        communicate: edge_tts.Communicate 实例
        on_chunk: 每拿到一个事件块时执行的回调
        timeout_seconds: 单次流式请求总超时；为 None 时不启用超时。
    """
    if hasattr(communicate, "stream_sync"):
        if timeout_seconds:
            _stream_edge_tts_sync_with_timeout(
                communicate, on_chunk, timeout_seconds
            )
            return

        for chunk in communicate.stream_sync():
            on_chunk(chunk)
        return

    if not hasattr(communicate, "stream"):
        raise AttributeError("edge_tts communicate object has no stream method")

    async def _consume_async_stream():
        async for chunk in communicate.stream():
            on_chunk(chunk)

    # 这里显式创建独立事件循环，而不是复用外部上下文，目的是避免
    # 在同步调用栈里遇到"当前线程没有事件循环"或跨线程复用循环的问题。
    loop = asyncio.new_event_loop()
    try:
        if timeout_seconds:
            loop.run_until_complete(
                asyncio.wait_for(_consume_async_stream(), timeout=timeout_seconds)
            )
        else:
            loop.run_until_complete(_consume_async_stream())
    finally:
        loop.close()


def azure_tts_v1(
    text: str, voice_name: str, voice_rate: float, voice_file: str
) -> Union[SubMaker, None]:
    voice_name = parse_voice_name(voice_name)
    text = text.strip()
    rate_str = convert_rate_to_percent(voice_rate)
    for i in range(3):
        try:
            logger.info(f"start, voice name: {voice_name}, try: {i + 1}")

            # 这里同时兼容 edge_tts 7.x 和可能残留的老依赖：
            # 1. 新版支持 `boundary` + `stream_sync()`
            # 2. 旧版不支持 `boundary`，且通常只暴露异步 `stream()`
            ensure_file_path_exists(voice_file)
            communicate = create_edge_tts_communicate(text, voice_name, rate_str)
            sub_maker = edge_tts.SubMaker()
            timeout_seconds = get_edge_tts_timeout_seconds()

            with open(voice_file, "wb") as file:
                def _handle_chunk(chunk):
                    chunk_type = chunk["type"]
                    if chunk_type == "audio":
                        file.write(chunk["data"])
                    elif chunk_type in ["WordBoundary", "SentenceBoundary"]:
                        # 无论来自 7.x 的同步流，还是旧版异步流，只要事件结构
                        # 里仍有边界信息，就统一喂给 SubMaker，保证后续字幕链路
                        # 仍然走项目现有逻辑。
                        sub_maker.feed(chunk)

                stream_edge_tts_chunks(
                    communicate, _handle_chunk, timeout_seconds=timeout_seconds
                )

            if not sub_maker.get_srt():
                logger.warning("failed, sub_maker.get_srt() is empty")
                continue

            logger.info(f"completed, output file: {voice_file}")
            return sub_maker
        except Exception as e:
            logger.error(f"failed, error: {str(e)}")
            # TTS 流式写入如果在首包前超时或网络异常，会留下 0 字节音频文件。
            # 这种文件既不可播放，也可能误导后续排查，因此失败后只清理空文件；
            # 如果已经写入了部分数据，则保留现场文件，便于分析服务端返回内容。
            if os.path.exists(voice_file) and os.path.getsize(voice_file) == 0:
                try:
                    os.remove(voice_file)
                except Exception as remove_error:
                    logger.warning(
                        "failed to remove empty tts file: "
                        f"{voice_file}, error: {str(remove_error)}"
                    )
    return None


def tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """v1 dispatcher: edge-tts only (the branch every other MPT TTS
    provider dispatch fell through to upstream)."""
    return azure_tts_v1(text, voice_name, voice_rate, voice_file)


def convert_rate_to_percent(rate: float) -> str:
    # edge-tts requires a sign-prefixed percentage (e.g. "+0%", "-20%").
    # Rounding can yield 0 for rates near but not equal to 1.0 (e.g. 1.004,
    # 0.997); those must still be returned as "+0%", not the unsigned "0%"
    # which edge-tts rejects with ValueError: Invalid rate '0%'.
    # API 或批处理调用可能传入 0、0.0、None 或无法转换的空值；这些值不代表
    # 合法语速，直接计算会变成 -100% 或抛异常。这里统一回退到正常语速，
    # 避免生成极慢音频或让 TTS 流程在边界输入下失败。
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        rate = 1.0
    if rate <= 0:
        rate = 1.0
    percent = round((rate - 1.0) * 100)
    if percent >= 0:
        return f"+{percent}%"
    return f"{percent}%"


def _format_text(text: str) -> str:
    """
    清理字幕对齐前的脚本文本。

    这里不能只在 LLM 生成阶段处理，因为用户也可能手动粘贴脚本，或通过
    API 直接传入包含 Markdown 标记的文本。TTS 通常不会朗读 `---`、
    `___`、`***` 这类分隔符行，也不会朗读 `_` 这种强调标记；如果字幕
    对齐仍保留这些字符，`create_subtitle()` 会一直等待不存在的 cue，
    最终导致字幕文件缺失并在 Whisper fallback 校正时补出全 0 时间轴。
    """
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    return utils.normalize_script_for_subtitle_matching(text)


def _build_subtitle_formatter():
    """
    返回统一的 SRT 行格式化函数。

    这里单独拆成一个小工具，是为了让 edge_tts 7.x 的 cues 路径
    和项目原有的 legacy `subs/offset` 路径共用同一套字幕落盘格式，
    避免两套逻辑各自产生细微格式差异。
    """

    def formatter(idx: int, start_time: float, end_time: float, sub_text: str) -> str:
        start_t = mktimestamp(start_time).replace(".", ",")
        end_t = mktimestamp(end_time).replace(".", ",")
        return f"{idx}\n{start_t} --> {end_t}\n{sub_text}\n"

    return formatter


# 阿拉伯语变音符号和 Tatweel 拉长符在 edge-tts 返回文本中可能出现，
# 这些字符不影响语义，但会导致脚本文本和字幕 cue 字符串精确匹配失败。
_ARABIC_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟـۖ-ۭ]")


def _normalize_arabic(text: str) -> str:
    """统一阿拉伯语常见字母变体，提升字幕 cue 与脚本行的匹配容错率。

    edge-tts 对阿拉伯语可能返回与原脚本不同的字母形态，例如把 أ/إ/آ
    归一成 ا，或者携带变音符号。这里仅在最后一层匹配兜底中使用，
    不改变原始字幕文本，避免影响最终展示内容。
    """
    text = _ARABIC_DIACRITICS.sub("", text)
    for src, dst in (
        ("أإآٱ", "ا"),
        ("ىئ", "ي"),
        ("ة", "ه"),
        ("ؤ", "و"),
    ):
        for ch in src:
            text = text.replace(ch, dst)
    return text


def _match_script_line(script_lines: list[str], current_text: str, sub_index: int) -> str:
    """
    尝试把当前累计的字幕文本，与脚本中的某一条标准断句匹配起来。

    1. 优先精确匹配；
    2. 再做一次去标点和 Markdown `_` 格式符后的匹配；
    3. 最后做一次阿拉伯语字符形态归一化匹配。

    这样可以兼容：
    - TTS 返回里可能缺失或单独拆分的标点；
    - 中文场景下词边界和脚本文本不完全一一对应的情况。
    """
    if len(script_lines) <= sub_index:
        return ""

    target_line = script_lines[sub_index]
    if current_text == target_line:
        return target_line.strip()

    current_text_normalized = re.sub(r"[_\W]+", "", current_text)
    target_line_normalized = re.sub(r"[_\W]+", "", target_line)
    if current_text_normalized == target_line_normalized:
        return target_line.strip()

    # 最后一层阿拉伯语容错：edge-tts 返回的字母形态、变音符号或 Tatweel
    # 可能和脚本不同。只在常规匹配失败后归一化比较，非阿拉伯语文本不会受影响。
    current_ar = re.sub(r"[_\W]+", "", _normalize_arabic(current_text))
    target_ar = re.sub(r"[_\W]+", "", _normalize_arabic(target_line))
    if current_ar and current_ar == target_ar:
        return target_line.strip()

    return ""


def _write_subtitle_items(sub_items: list[str], subtitle_file: str) -> bool:
    """
    将已经聚合好的字幕段写入到 SRT 文件，并做一次基本可读性验证。

    返回值：
    - `True`：字幕文件成功落盘且可被 moviepy 解析；
    - `False`：字幕文件写入或解析失败。
    """
    try:
        ensure_file_path_exists(subtitle_file)
        with open(subtitle_file, "w", encoding="utf-8") as file:
            file.write("\n".join(sub_items) + "\n")

        sbs = subtitles.file_to_subtitles(subtitle_file, encoding="utf-8")
        duration = max([tb for ((ta, tb), txt) in sbs]) if sbs else 0
        logger.info(
            f"completed, subtitle file created: {subtitle_file}, duration: {duration}"
        )
        return True
    except Exception as e:
        logger.error(f"failed, error: {str(e)}")
        if os.path.exists(subtitle_file):
            os.remove(subtitle_file)
        return False


def _build_subtitle_items_from_edge_cues(
    sub_maker: SubMaker, script_lines: list[str]
) -> list[str]:
    """
    将 edge_tts 7.x 的细粒度 `cues` 聚合为按脚本断句的 SRT 片段。

    背景：
    edge_tts 7.x 的 `SubMaker.get_srt()` 更偏向逐词/逐短语的时间轴。
    对英文做逐词高亮尚可，但中文短视频字幕如果直接照搬，会出现
    "金钱 / 是 / 一种 / 社会 / 工具" 这种阅读体验很差的效果。

    实现策略：
    1. 逐个消费 cues 中的 `content`；
    2. 累积成一段候选文本；
    3. 当候选文本与脚本里当前目标断句匹配时，收敛为一个完整字幕段；
    4. 使用第一条 cue 的开始时间和最后一条 cue 的结束时间，保证时间轴连续。
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    current_text = ""
    current_start_time = None

    for cue in sub_maker.cues:
        cue_text = unescape(cue.content)
        if current_start_time is None:
            current_start_time = int(cue.start.total_seconds() * 10000000)

        current_end_time = int(cue.end.total_seconds() * 10000000)
        current_text += cue_text

        matched_text = _match_script_line(script_lines, current_text, sub_index)
        if not matched_text:
            continue

        sub_index += 1
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=current_start_time,
                end_time=current_end_time,
                sub_text=matched_text,
            )
        )
        current_text = ""
        current_start_time = None

    if current_text.strip():
        logger.warning(
            f"edge cues still have unmatched text after aggregation: {current_text}"
        )

    return sub_items


def _build_subtitle_items_from_legacy_submaker(
    sub_maker: SubMaker, script_lines: list[str]
) -> list[str]:
    """
    将 `subs/offset` 结构聚合为按脚本断句的 SRT 片段。

    edge-tts 的 SubMaker in practice always exposes ``cues`` in v1 (the
    branch this function serves is the pre-7.x fallback), but it is kept
    since ``create_subtitle`` dispatches on ``hasattr(sub_maker, "cues")``.
    """
    formatter = _build_subtitle_formatter()
    start_time = -1.0
    sub_items = []
    sub_index = 0
    sub_line = ""

    legacy_offsets = getattr(sub_maker, "offset", [])
    legacy_subs = getattr(sub_maker, "subs", [])
    for _, (offset, sub) in enumerate(zip(legacy_offsets, legacy_subs)):
        current_start_time, current_end_time = offset
        if start_time < 0:
            start_time = current_start_time

        sub_line += unescape(sub)
        matched_text = _match_script_line(script_lines, sub_line, sub_index)
        if not matched_text:
            continue

        sub_index += 1
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=current_end_time,
                sub_text=matched_text,
            )
        )
        start_time = -1.0
        sub_line = ""

    if sub_line.strip():
        logger.warning(
            f"legacy subtitle items still have unmatched text after aggregation: {sub_line}"
        )

    return sub_items


def create_subtitle(sub_maker: SubMaker, text: str, subtitle_file: str):
    """
    优化字幕文件
    1. 将字幕文件按照标点符号分割成多行
    2. 逐行匹配字幕文件中的文本
    3. 生成新的字幕文件
    """
    text = _format_text(text)
    script_lines = utils.split_string_by_punctuations(text)
    try:
        if hasattr(sub_maker, "cues") and sub_maker.cues:
            sub_items = _build_subtitle_items_from_edge_cues(sub_maker, script_lines)
        else:
            sub_items = _build_subtitle_items_from_legacy_submaker(
                sub_maker, script_lines
            )

        if len(sub_items) != len(script_lines):
            logger.warning(
                f"failed, sub_items len: {len(sub_items)}, script_lines len: {len(script_lines)}"
            )
            return

        _write_subtitle_items(sub_items, subtitle_file)
    except Exception as e:
        logger.error(f"failed, error: {str(e)}")


def _get_audio_duration_from_submaker(sub_maker: SubMaker):
    """
    获取音频时长
    """
    # 优先兼容 edge_tts 7.x 的 cues 结构；
    # 如果是手工填充的旧结构，则继续读取 offset。
    if hasattr(sub_maker, "cues") and sub_maker.cues:
        return sub_maker.cues[-1].end.total_seconds()

    legacy_offsets = getattr(sub_maker, "offset", [])
    if not legacy_offsets:
        return 0.0
    return legacy_offsets[-1][1] / 10000000


def _get_audio_duration_from_file(audio_file: str) -> float:
    """
    获取音频文件时长（支持 mp3/m4a/wav/aac 等 ffmpeg 可解码的格式）
    """
    if not os.path.exists(audio_file):
        logger.error(f"audio file does not exist: {audio_file}")
        return 0.0

    try:
        # Use moviepy (ffmpeg) to read the duration of any supported audio format
        with AudioFileClip(audio_file) as audio:
            return audio.duration  # Duration in seconds
    except Exception as e:
        logger.error(f"Failed to get audio duration from file: {str(e)}")
        return 0.0


def get_audio_duration(target: Union[str, SubMaker]) -> float:
    """
    获取音频时长
    如果是SubMaker对象，则从SubMaker中获取时长
    如果是音频文件路径，则从音频文件中获取时长（支持 mp3/m4a/wav 等格式）
    """
    if isinstance(target, SubMaker):
        return _get_audio_duration_from_submaker(target)
    elif isinstance(target, str):
        return _get_audio_duration_from_file(target)
    else:
        logger.error(f"Invalid target type: {type(target)}")
        return 0.0
