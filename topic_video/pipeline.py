"""Topic-to-Video orchestrator.

Ports MoneyPrinterTurbo's ``app/services/task.py`` pipeline (script -> terms
-> audio -> subtitle -> materials -> video) down to the v1 slice OpenShorts
supports: OpenAI-only script/terms generation, edge-tts-only narration,
Pexels-only stock footage, MoviePy composition with no BGM. Follows the same
shape as ``saasshorts.py``'s ``generate_full_video()``: a plain function of
``(topic, config, output_dir, log)`` that FastAPI wraps in a job and runs via
``run_in_executor`` (see openshorts/app.py's ``/api/topicvideo/generate``).

Concurrency: this module (specifically ``generate_topic_video``) is the
function that ``run_in_executor`` runs on a dedicated worker thread per job.
The very first thing it does is populate ``mpt_config.app`` for *this
thread* -- required because voice.py/material.py/llm.py all read
``config.app.get(...)`` with no per-call override, and ``mpt_config.app`` is
threading.local()-backed specifically so concurrent jobs never see each
other's OpenAI/Pexels keys (see mpt_config.py's docstring).
"""

import json
import math
import os
import re
from typing import Callable, Dict, List

from . import mpt_config
from . import llm
from . import voice
from . import material
from . import video
from .schema import VideoAspect, VideoConcatMode, VideoParams


class TopicVideoError(Exception):
    """Raised when a pipeline stage fails. ``stage`` names which of the 6
    stages failed, matching the vocabulary MPT's own task.py uses
    (script/terms/audio/subtitle/materials/video) so the job status API can
    report a ``failed_stage`` the same way ``/api/saasshorts/status`` and
    the clip generator's ``/api/status`` already do."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


# A curated slice of edge-tts voices covering the languages the v1 UI
# actually offers. edge-tts has hundreds of voices; MPT's own WebUI fetches
# the full list from the network on every load; a fixed curated list is
# instant and avoids a network round-trip for something this pipeline can
# only use one of anyway. Exposed via GET /api/topicvideo/voices.
CURATED_VOICES = [
    {"name": "English (US) - Aria (Female)", "voice_name": "en-US-AriaNeural-Female", "language": "en"},
    {"name": "English (US) - Guy (Male)", "voice_name": "en-US-GuyNeural-Male", "language": "en"},
    {"name": "English (UK) - Sonia (Female)", "voice_name": "en-GB-SoniaNeural-Female", "language": "en"},
    {"name": "Spanish (ES) - Elvira (Female)", "voice_name": "es-ES-ElviraNeural-Female", "language": "es"},
    {"name": "Spanish (MX) - Dalia (Female)", "voice_name": "es-MX-DaliaNeural-Female", "language": "es"},
    {"name": "Portuguese (BR) - Francisca (Female)", "voice_name": "pt-BR-FranciscaNeural-Female", "language": "pt"},
    {"name": "Portuguese (PT) - Raquel (Female)", "voice_name": "pt-PT-RaquelNeural-Female", "language": "pt"},
    {"name": "French (FR) - Denise (Female)", "voice_name": "fr-FR-DeniseNeural-Female", "language": "fr"},
    {"name": "German (DE) - Katja (Female)", "voice_name": "de-DE-KatjaNeural-Female", "language": "de"},
    {"name": "Italian (IT) - Elsa (Female)", "voice_name": "it-IT-ElsaNeural-Female", "language": "it"},
]
DEFAULT_VOICE_NAME = CURATED_VOICES[0]["voice_name"]


def _exists(path: str) -> bool:
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:max_len] or "topic"


def _build_params(topic: str, config: Dict) -> VideoParams:
    aspect = config.get("video_aspect") or VideoAspect.portrait.value
    voice_name = config.get("voice_name") or DEFAULT_VOICE_NAME
    return VideoParams(
        video_subject=topic,
        video_aspect=aspect,
        video_concat_mode=VideoConcatMode.random.value,
        video_transition_mode=None,
        video_clip_duration=int(config.get("video_clip_duration", 5)),
        video_clip_speed=1.0,
        video_count=1,
        video_source="pexels",
        video_language=config.get("video_language", ""),
        voice_name=voice_name,
        voice_volume=1.0,
        voice_rate=float(config.get("voice_rate", 1.0)),
        # v1 scope has no BGM provider; an empty bgm_type makes
        # video.generate_video() skip the BGM branch entirely (see
        # bgm.should_use_bgm(), which returns False for a falsy bgm_type).
        bgm_type="",
        bgm_file="",
        bgm_volume=0.0,
        subtitle_enabled=bool(config.get("subtitle_enabled", True)),
        # NotoSerif-Bold.ttf is what actually ships in openshorts/fonts/ --
        # MPT's own default (STHeitiMedium.ttc, a CJK font) does not exist in
        # this repo. topic_video/utils.py's font_dir() override points at
        # openshorts/fonts/, so this name must match a real file there.
        font_name="NotoSerif-Bold.ttf",
        text_fore_color="#FFFFFF",
        stroke_color="#000000",
        stroke_width=1.5,
        font_size=int(config.get("font_size", 60)),
        n_threads=2,
        paragraph_number=int(config.get("paragraph_number", 1)),
    )


def _configure_thread_local_app(config: Dict, materials_dir: str) -> None:
    """Populate mpt_config.app for the current thread only. Must run before
    any of llm.py / voice.py / material.py touch config.app -- see the
    module docstring and mpt_config.py's threading.local rationale."""
    openai_key = (config.get("openai_api_key") or "").strip()
    pexels_key = (config.get("pexels_api_key") or "").strip()
    if not openai_key:
        raise TopicVideoError("script", "missing OpenAI API key")
    if not pexels_key:
        raise TopicVideoError("materials", "missing Pexels API key")

    mpt_config.app.clear()
    mpt_config.app.update(
        {
            "llm_provider": "openai",
            "openai_api_key": openai_key,
            "openai_model_name": config.get("openai_model") or "",
            "openai_base_url": config.get("openai_base_url") or "",
            "pexels_api_keys": pexels_key,
            # Confines this job's downloaded stock clips to its own output
            # directory instead of MPT's process-wide storage/cache_videos
            # cache. See material.py's download_videos(): it falls back to
            # the shared cache dir when this is unset or not a directory,
            # so `materials_dir` must already exist by the time this runs.
            "material_directory": materials_dir,
        }
    )


def _stage_script(topic: str, output_dir: str, params: VideoParams, log: Callable[[str], None]) -> str:
    script_path = os.path.join(output_dir, "script.txt")
    if _exists(script_path):
        log("[1/6] script: using cached script.txt")
        with open(script_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    log("[1/6] script: generating narration script with OpenAI")
    script = llm.generate_script(
        video_subject=topic,
        language=params.video_language,
        paragraph_number=params.paragraph_number,
    )
    if not script or script.startswith("Error:") or script.startswith("Error "):
        raise TopicVideoError("script", script or "failed to generate video script")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    log("[1/6] script: ready")
    return script


def _stage_terms(
    topic: str, script: str, output_dir: str, log: Callable[[str], None]
) -> List[str]:
    terms_path = os.path.join(output_dir, "terms.json")
    if _exists(terms_path):
        log("[2/6] terms: using cached terms.json")
        with open(terms_path, "r", encoding="utf-8") as f:
            return json.load(f)

    log("[2/6] terms: generating stock-footage search terms")
    terms = llm.generate_terms(video_subject=topic, video_script=script, amount=5)
    if not terms:
        raise TopicVideoError("terms", "failed to generate video search terms")

    with open(terms_path, "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)
    log(f"[2/6] terms: ready ({', '.join(terms)})")
    return terms


def _stage_audio(
    script: str, output_dir: str, params: VideoParams, log: Callable[[str], None]
):
    """Returns (audio_file, sub_maker_or_none). sub_maker is None when the
    audio file was already cached from a previous run -- SubMaker objects
    aren't persisted to disk, only the rendered audio is."""
    audio_file = os.path.join(output_dir, "audio.mp3")
    if _exists(audio_file):
        log("[3/6] audio: using cached audio.mp3")
        return audio_file, None

    log("[3/6] audio: synthesizing narration with edge-tts")
    sub_maker = voice.tts(
        text=script,
        voice_name=voice.parse_voice_name(params.voice_name),
        voice_rate=params.voice_rate,
        voice_file=audio_file,
    )
    if sub_maker is None:
        raise TopicVideoError(
            "audio",
            "failed to synthesize narration audio; verify the selected voice "
            "and edge-tts connectivity",
        )
    log("[3/6] audio: ready")
    return audio_file, sub_maker


def _stage_subtitle(
    script: str,
    output_dir: str,
    params: VideoParams,
    audio_file: str,
    sub_maker,
    log: Callable[[str], None],
) -> str:
    if not params.subtitle_enabled:
        log("[4/6] subtitle: disabled, skipping")
        return ""

    subtitle_path = os.path.join(output_dir, "subtitle.srt")
    if _exists(subtitle_path):
        log("[4/6] subtitle: using cached subtitle.srt")
        return subtitle_path

    if sub_maker is None:
        # audio.mp3 came from cache (previous run), so there's no in-memory
        # SubMaker to align against the script. edge-tts is free, so just
        # resynthesize to recover subtitle timing rather than shipping the
        # video without subtitles.
        log("[4/6] subtitle: re-synthesizing audio to recover subtitle timing")
        sub_maker = voice.tts(
            text=script,
            voice_name=voice.parse_voice_name(params.voice_name),
            voice_rate=params.voice_rate,
            voice_file=audio_file,
        )
        if sub_maker is None:
            raise TopicVideoError(
                "subtitle", "failed to synthesize audio for subtitle timing"
            )

    log("[4/6] subtitle: aligning subtitle to script")
    voice.create_subtitle(sub_maker=sub_maker, text=script, subtitle_file=subtitle_path)
    if not _exists(subtitle_path):
        log("[4/6] subtitle: alignment failed, continuing without subtitles")
        return ""
    log("[4/6] subtitle: ready")
    return subtitle_path


def _stage_materials(
    task_id: str,
    terms: List[str],
    audio_duration: float,
    output_dir: str,
    params: VideoParams,
    log: Callable[[str], None],
) -> List[str]:
    materials_dir = os.path.join(output_dir, "materials")
    os.makedirs(materials_dir, exist_ok=True)
    manifest_path = os.path.join(materials_dir, "manifest.json")

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                cached_paths = json.load(f)
            if cached_paths and all(_exists(p) for p in cached_paths):
                log(f"[5/6] materials: using {len(cached_paths)} cached clips")
                return cached_paths
        except (OSError, ValueError):
            pass  # fall through and re-download

    log(f"[5/6] materials: searching and downloading Pexels clips for {len(terms)} terms")
    video_paths = material.download_videos(
        task_id=task_id,
        search_terms=terms,
        source="pexels",
        video_aspect=params.video_aspect,
        video_concat_mode=params.video_concat_mode,
        audio_duration=audio_duration * params.video_count,
        max_clip_duration=params.video_clip_duration,
    )
    if not video_paths:
        raise TopicVideoError(
            "materials", "failed to download video materials from Pexels"
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(video_paths, f)
    log(f"[5/6] materials: downloaded {len(video_paths)} clips")
    return video_paths


def _stage_video(
    video_paths: List[str],
    audio_file: str,
    subtitle_path: str,
    output_dir: str,
    params: VideoParams,
    log: Callable[[str], None],
) -> str:
    final_video_path = os.path.join(output_dir, "final.mp4")
    if _exists(final_video_path):
        log("[6/6] video: using cached final.mp4")
        return final_video_path

    combined_video_path = os.path.join(output_dir, "combined.mp4")
    log("[6/6] video: combining source clips")
    video.combine_videos(
        combined_video_path=combined_video_path,
        video_paths=video_paths,
        audio_file=audio_file,
        video_aspect=params.video_aspect,
        video_concat_mode=params.video_concat_mode,
        video_transition_mode=params.video_transition_mode,
        max_clip_duration=params.video_clip_duration,
        threads=params.n_threads,
        clip_speed=params.video_clip_speed,
    )

    log("[6/6] video: rendering final video (narration + subtitles)")
    video.generate_video(
        video_path=combined_video_path,
        audio_path=audio_file,
        subtitle_path=subtitle_path,
        output_file=final_video_path,
        params=params,
    )
    if not _exists(final_video_path):
        raise TopicVideoError("video", "video generation did not produce an output file")
    log("[6/6] video: ready")
    return final_video_path


def generate_topic_video(
    topic: str,
    config: Dict,
    output_dir: str,
    log: Callable[[str], None] = print,
) -> Dict:
    """
    Run the full Topic-to-Video pipeline.

    Args:
        topic: the video subject / prompt.
        config: {
            "openai_api_key": str (required, BYOK),
            "pexels_api_key": str (required, BYOK),
            "openai_model": str (optional, defaults to the OpenAI provider's
                default model),
            "openai_base_url": str (optional),
            "video_language": str (optional, "" = auto-detect),
            "voice_name": str (optional edge-tts voice id, see
                CURATED_VOICES; defaults to DEFAULT_VOICE_NAME),
            "voice_rate": float (optional, default 1.0),
            "video_aspect": str (optional, "16:9" | "9:16" | "1:1",
                default "9:16"),
            "subtitle_enabled": bool (optional, default True),
        }
        output_dir: directory to write output files to (created if missing).
        log: progress log callback.

    Returns:
        {"video_path": str, "video_filename": str, "subtitle_path": str,
         "script": str, "terms": list[str], "duration": float}

    Raises:
        TopicVideoError: a stage failed. ``.stage`` names which one.
    """
    os.makedirs(output_dir, exist_ok=True)
    materials_dir = os.path.join(output_dir, "materials")
    os.makedirs(materials_dir, exist_ok=True)

    # Must happen before any ported module reads config.app -- see module
    # docstring and mpt_config.py.
    _configure_thread_local_app(config, materials_dir)

    task_id = os.path.basename(os.path.normpath(output_dir))
    params = _build_params(topic, config)

    script = _stage_script(topic, output_dir, params, log)
    terms = _stage_terms(topic, script, output_dir, log)
    audio_file, sub_maker = _stage_audio(script, output_dir, params, log)
    audio_duration = math.ceil(voice.get_audio_duration(audio_file))
    if audio_duration <= 0:
        raise TopicVideoError("audio", "generated audio duration is zero")
    subtitle_path = _stage_subtitle(
        script, output_dir, params, audio_file, sub_maker, log
    )
    video_paths = _stage_materials(
        task_id, terms, audio_duration, output_dir, params, log
    )
    final_video_path = _stage_video(
        video_paths, audio_file, subtitle_path, output_dir, params, log
    )

    return {
        "video_path": final_video_path,
        "video_filename": os.path.basename(final_video_path),
        "subtitle_path": subtitle_path,
        "script": script,
        "terms": terms,
        "duration": audio_duration,
    }
