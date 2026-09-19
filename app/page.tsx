'use client';

import {
  Captions,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  FileVideo,
  Highlighter,
  LoaderCircle,
  Pause,
  Play,
  Repeat2,
  RotateCcw,
  Sparkles,
  Upload,
  Volume2,
  WandSparkles,
  X,
} from 'lucide-react';
import { ChangeEvent, CSSProperties, DragEvent, useEffect, useMemo, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import {
  NativeSelect,
  NativeSelectOption,
} from '@/components/ui/native-select';
import {
  Progress,
  ProgressLabel,
} from '@/components/ui/progress';

type Mode = 'english' | 'chinese' | 'bilingual' | 'cloze';
type Highlight = 'none' | 'mint' | 'orange' | 'blue';

type CaptionWord = {
  text: string;
  start: number;
  end: number;
  highlight: Highlight;
};

type Caption = {
  id: number;
  start: number;
  end: number;
  english: string;
  chinese: string;
  words: CaptionWord[];
  sourceWords?: CaptionWord[];
  sourceLanguage?: 'en' | 'zh';
};

type StreamEvent =
  | { type: 'status'; message: string; progress?: number; model?: string }
  | { type: 'segment'; caption: Caption }
  | { type: 'translation'; id: number; english: string; chinese: string; words?: CaptionWord[] }
  | { type: 'done'; message: string; count: number; model?: string }
  | { type: 'error'; message: string };

const LOCAL_ASR_URL = 'http://127.0.0.1:8766';

const DEMO_CAPTIONS: Caption[] = [
  {
    id: 1,
    start: 0,
    end: 5.2,
    english: 'Learning a language is about finding your own rhythm.',
    chinese: '学习一门语言，就是找到属于你自己的节奏。',
    words: wordify('Learning a language is about finding your own rhythm.', 0, 5.2, [0, 2, 5, 7]),
  },
  {
    id: 2,
    start: 5.2,
    end: 10.1,
    english: 'Listen closely, then say it in your own way.',
    chinese: '仔细听，然后用你自己的方式说出来。',
    words: wordify('Listen closely, then say it in your own way.', 5.2, 10.1, [0, 1, 5]),
  },
  {
    id: 3,
    start: 10.1,
    end: 15.7,
    english: 'Small moments of practice add up to real confidence.',
    chinese: '每一次小小的练习，都会积累成真正的自信。',
    words: wordify('Small moments of practice add up to real confidence.', 10.1, 15.7, [1, 3, 8]),
  },
  {
    id: 4,
    start: 15.7,
    end: 21.4,
    english: 'The goal is not perfection, but clear communication.',
    chinese: '目标不是完美，而是清晰地表达。',
    words: wordify('The goal is not perfection, but clear communication.', 15.7, 21.4, [4, 6, 7]),
  },
];

const MODE_OPTIONS: { value: Mode; label: string; note: string }[] = [
  { value: 'english', label: '英文', note: '专注原声' },
  { value: 'chinese', label: '中文', note: '理解语义' },
  { value: 'bilingual', label: '双语', note: '对照学习' },
  { value: 'cloze', label: '挖空', note: '主动回忆' },
];

const HIGHLIGHT_CLASSES: Record<Highlight, string> = {
  none: '',
  mint: 'word-mint',
  orange: 'word-orange',
  blue: 'word-blue',
};

function wordify(text: string, start: number, end: number, accented: number[] = []): CaptionWord[] {
  const parts = text.trim().split(/\s+/);
  const unit = (end - start) / Math.max(parts.length, 1);
  return parts.map((word, index) => ({
    text: word,
    start: start + unit * index,
    end: start + unit * (index + 1),
    highlight: accented.includes(index)
      ? index % 3 === 0
        ? 'orange'
        : index % 3 === 1
          ? 'blue'
          : 'mint'
      : 'none',
  }));
}

function formatTime(seconds: number) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const rest = Math.floor(safe % 60);
  return `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}`;
}

function fileSize(bytes: number) {
  return bytes < 1024 * 1024
    ? `${Math.ceil(bytes / 1024)} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const captionRefs = useRef<Array<HTMLElement | null>>([]);
  const objectUrlRef = useRef<string | null>(null);
  const recognitionAbortRef = useRef<AbortController | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [captions, setCaptions] = useState<Caption[]>(DEMO_CAPTIONS);
  const [mode, setMode] = useState<Mode>('bilingual');
  const [activeIndex, setActiveIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRateState] = useState(1);
  const [loopSentence, setLoopSentence] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [notice, setNotice] = useState('演示字幕已加载；导入视频后可自动生成你的课程。');
  const [error, setError] = useState('');
  const [recognizer, setRecognizer] = useState<'checking' | 'local' | 'cloud'>('checking');
  const [recognizerLabel, setRecognizerLabel] = useState('正在检查本地模型…');

  const activeCaption = captions[activeIndex] ?? captions[0];
  const hasOwnLesson = Boolean(file);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${LOCAL_ASR_URL}/health`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('local asr unavailable');
        const payload = (await response.json()) as { ok?: boolean; model?: string };
        if (!payload.ok) throw new Error('local model missing');
        setRecognizer('local');
        setRecognizerLabel(payload.model || 'Qwen3-ASR-0.6B + ForcedAligner-0.6B');
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setRecognizer('cloud');
          setRecognizerLabel('本地服务未启动 · 将尝试云端');
        }
      });
    return () => {
      controller.abort();
      recognitionAbortRef.current?.abort();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  useEffect(() => {
    captionRefs.current[activeIndex]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [activeIndex]);

  useEffect(() => {
    if (!isPlaying) return;
    let frame = 0;
    const followPlayback = () => {
      const video = videoRef.current;
      if (video) {
        const time = video.currentTime;
        setCurrentTime(time);
        const index = captions.findIndex((caption) => time >= caption.start && time < caption.end);
        if (index >= 0) setActiveIndex((current) => current === index ? current : index);
      }
      frame = requestAnimationFrame(followPlayback);
    };
    frame = requestAnimationFrame(followPlayback);
    return () => cancelAnimationFrame(frame);
  }, [isPlaying, captions]);

  const progressPercent = useMemo(() => {
    if (!duration) return 0;
    return Math.min(100, (currentTime / duration) * 100);
  }, [currentTime, duration]);

  function loadFile(nextFile: File) {
    setError('');
    if (!nextFile.type.startsWith('video/') && !/\.(mp4|mov|m4v|webm)$/i.test(nextFile.name)) {
      setError('请选择 MP4、MOV、M4V 或 WEBM 视频。');
      return;
    }
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const nextUrl = URL.createObjectURL(nextFile);
    objectUrlRef.current = nextUrl;
    setVideoUrl(nextUrl);
    setFile(nextFile);
    setCaptions([]);
    setActiveIndex(0);
    setCurrentTime(0);
    setNotice('视频已加载，正在自动启动本地识别…');
    void generateLesson(nextFile);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0];
    if (nextFile) loadFile(nextFile);
    event.target.value = '';
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const nextFile = event.dataTransfer.files?.[0];
    if (nextFile) loadFile(nextFile);
  }

  async function generateLesson(selectedFile?: File) {
    const sourceFile = selectedFile ?? file;
    if (!sourceFile) return;
    recognitionAbortRef.current?.abort();
    const controller = new AbortController();
    recognitionAbortRef.current = controller;
    setProcessing(true);
    setError('');
    setCaptions([]);
    setActiveIndex(0);
    setProgress(1);
    setNotice('正在把视频交给本地 Qwen3-ASR…');

    try {
      await streamLocalTranscription(sourceFile, controller.signal);
      if (videoRef.current) videoRef.current.currentTime = 0;
    } catch (reason) {
      if (controller.signal.aborted) return;
      try {
        setRecognizer('cloud');
        setRecognizerLabel('本地服务未连接 · 正在尝试云端');
        setNotice('本地 Qwen3-ASR 服务未连接，正在尝试云端识别…');
        await processWithCloud(sourceFile, controller.signal);
      } catch (fallbackReason) {
        if (controller.signal.aborted) return;
        setError(fallbackReason instanceof Error ? fallbackReason.message : '生成失败，请稍后再试。');
        setNotice('视频已保留，可以启动本地服务后点击“重新识别”。');
      }
    } finally {
      if (!controller.signal.aborted) setProcessing(false);
    }
  }

  async function streamLocalTranscription(sourceFile: File, signal: AbortSignal) {
    const response = await fetch(`${LOCAL_ASR_URL}/transcribe`, {
      method: 'POST',
      headers: {
        'Content-Type': sourceFile.type || 'application/octet-stream',
        'X-File-Name': encodeURIComponent(sourceFile.name),
      },
      body: sourceFile,
      signal,
    });
    if (!response.ok || !response.body) {
      const payload = (await response.json().catch(() => ({}))) as { error?: string };
          throw new Error(payload.error || '本地 Qwen3-ASR 服务没有响应。');
    }

    setRecognizer('local');
      setRecognizerLabel('Qwen3-ASR-0.6B + ForcedAligner-0.6B');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let received = 0;

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        if (!line.trim()) continue;
        const message = JSON.parse(line) as StreamEvent;
        if (message.type === 'status') {
          setNotice(message.message);
          if (typeof message.progress === 'number') setProgress(message.progress);
          if (message.model) setRecognizerLabel(message.model);
        } else if (message.type === 'segment') {
          received += 1;
          setCaptions((items) => [...items, message.caption]);
          setNotice(`正在流式生成字幕… 已识别 ${received} 句`);
        } else if (message.type === 'translation') {
          setCaptions((items) => items.map((caption) => caption.id === message.id
            ? { ...caption, english: message.english, chinese: message.chinese, words: message.words?.length ? message.words : caption.words }
            : caption));
        } else if (message.type === 'done') {
          setProgress(100);
          setNotice(message.message);
        } else if (message.type === 'error') {
          throw new Error(message.message);
        }
      }
      if (done) break;
    }
    if (!received) throw new Error('没有识别到可用的中文或英文语音。');
  }

  async function processWithCloud(sourceFile: File, signal: AbortSignal) {
    if (sourceFile.size > 25 * 1024 * 1024) {
      throw new Error('本地 Qwen3-ASR 服务未启动；云端备用接口仅支持 25 MB，请运行 pnpm dev 后重试。');
    }
    const body = new FormData();
    body.append('file', sourceFile);
    const response = await fetch('/api/process', { method: 'POST', body, signal });
    const payload = (await response.json()) as { captions?: Caption[]; error?: string };
    if (!response.ok || !payload.captions?.length) throw new Error(payload.error || '没有识别到可用的中文或英文语音。');
    setCaptions(payload.captions);
    setActiveIndex(0);
    setProgress(100);
    setNotice(`课程已生成：${payload.captions.length} 句。`);
  }

  function onTimeUpdate() {
    const video = videoRef.current;
    if (!video) return;
    const time = video.currentTime;
    setCurrentTime(time);
    const index = captions.findIndex((caption) => time >= caption.start && time < caption.end);
    if (index >= 0 && index !== activeIndex) setActiveIndex(index);
    if (loopSentence && activeCaption && time >= activeCaption.end - 0.04) {
      video.currentTime = activeCaption.start;
      void video.play();
    }
  }

  function seekTo(index: number, autoPlay = true) {
    const next = captions[index];
    if (!next) return;
    setActiveIndex(index);
    setCurrentTime(next.start);
    if (videoRef.current && videoUrl) {
      videoRef.current.currentTime = next.start;
      if (autoPlay) void videoRef.current.play();
    }
  }

  function togglePlay() {
    const video = videoRef.current;
    if (!video || !videoUrl) {
      if (captions.length) seekTo((activeIndex + 1) % captions.length, false);
      return;
    }
    if (video.paused) void video.play();
    else video.pause();
  }

  function setPlaybackRate(value: string) {
    const rate = Number(value);
    setPlaybackRateState(rate);
    if (videoRef.current) videoRef.current.playbackRate = rate;
  }

  function seekVideo(value: string) {
    const video = videoRef.current;
    if (!video) return;
    const nextTime = Math.min(Math.max(Number(value), 0), duration || video.duration || 0);
    video.currentTime = nextTime;
    setCurrentTime(nextTime);
    const index = captions.findIndex((caption) => nextTime >= caption.start && nextTime < caption.end);
    if (index >= 0) setActiveIndex(index);
  }

  function cycleHighlight(captionIndex: number, wordIndex: number) {
    const order: Highlight[] = ['none', 'mint', 'orange', 'blue'];
    setCaptions((items) =>
      items.map((caption, index) => {
        if (index !== captionIndex) return caption;
        return {
          ...caption,
          words: caption.words.map((word, innerIndex) =>
            innerIndex === wordIndex
              ? { ...word, highlight: order[(order.indexOf(word.highlight) + 1) % order.length] }
              : word,
          ),
        };
      }),
    );
  }

  function clearVideo() {
    recognitionAbortRef.current?.abort();
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
    setFile(null);
    setVideoUrl('');
    setCaptions(DEMO_CAPTIONS);
    setActiveIndex(0);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setError('');
    setProgress(0);
    setProcessing(false);
    setNotice('已返回演示课程。');
  }

  return (
    <main className="min-h-screen bg-[#f4f1e9] text-[#17201d]">
      <header className="mx-auto flex h-[76px] max-w-[1500px] items-center justify-between px-4 sm:px-7 lg:px-10">
        <div className="flex items-center gap-3">
          <span className="grid size-9 place-items-center rounded-xl bg-[#163f35] text-[#f8f0d8]">
            <Captions size={19} strokeWidth={2.4} />
          </span>
          <div>
            <p className="text-lg font-semibold leading-none tracking-[-0.04em]">FluentFrame</p>
            <p className="mt-1 hidden text-[10px] font-semibold uppercase tracking-[.18em] text-[#8b948f] sm:block">Video language studio</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="hidden items-center gap-2 rounded-full border border-black/8 bg-white/65 px-3.5 py-2 text-xs text-[#59635f] md:flex">
            <span className={`size-1.5 rounded-full ${recognizer === 'local' ? 'bg-[#54b78f]' : recognizer === 'checking' ? 'animate-pulse bg-[#e6ad50]' : 'bg-[#d27a5e]'}`} />
            {recognizerLabel}
          </span>
          {hasOwnLesson && (
            <>
              <Button className="rounded-full bg-white" onClick={() => fileInputRef.current?.click()} size="sm" variant="outline">
                <FileVideo data-icon="inline-start" />
                重新选择视频
              </Button>
              <Button aria-label="关闭当前视频" className="rounded-full bg-white" onClick={clearVideo} size="icon-lg" variant="outline">
                <X />
              </Button>
            </>
          )}
        </div>
      </header>

      <section className="mx-auto grid max-w-[1500px] gap-4 px-3 pb-5 sm:px-6 lg:grid-cols-[minmax(0,1.08fr)_minmax(430px,.92fr)] lg:px-8">
        <section className="flex min-h-[470px] flex-col overflow-hidden rounded-[26px] bg-[#122a25] shadow-[0_24px_70px_rgba(20,41,35,.16)] lg:h-[calc(100vh-98px)] lg:min-h-[630px]">
          <div
            className={`relative flex flex-1 overflow-hidden ${dragging ? 'ring-4 ring-inset ring-[#efbd63]' : ''}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={(event) => { if (event.currentTarget === event.target) setDragging(false); }}
            onDragOver={(event) => event.preventDefault()}
            onDrop={onDrop}
          >
            {videoUrl ? (
              <video
                ref={videoRef}
                className="h-full min-h-[360px] w-full cursor-pointer bg-black object-contain"
                onClick={togglePlay}
                onDurationChange={(event) => setDuration(event.currentTarget.duration || 0)}
                onEnded={() => setIsPlaying(false)}
                onLoadedMetadata={(event) => { event.currentTarget.playbackRate = playbackRate; }}
                onPause={() => setIsPlaying(false)}
                onPlay={() => setIsPlaying(true)}
                onTimeUpdate={onTimeUpdate}
                playsInline
                src={videoUrl}
              />
            ) : (
              <div className="relative grid min-h-[390px] w-full place-items-center overflow-hidden px-7 text-center">
                <div className="video-aurora absolute inset-0" />
                <div className="relative max-w-md">
                  <span className="mx-auto mb-5 grid size-14 place-items-center rounded-2xl border border-white/15 bg-white/10 text-[#dff5e8] backdrop-blur">
                    <Upload size={23} />
                  </span>
                  <h1 className="font-serif text-3xl leading-tight tracking-[-0.035em] text-white sm:text-4xl">
                    把视频变成你的英语课堂
                  </h1>
                  <p className="mx-auto mt-4 max-w-sm text-sm leading-6 text-white/60">
                    拖入带有英文语音的视频，自动生成逐句字幕、中文翻译和单词时间轴。
                  </p>
                  <Button
                    className="mt-7 h-11 rounded-full bg-[#efbd63] px-5 font-semibold text-[#252119] hover:bg-[#f5cb7d]"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <FileVideo data-icon="inline-start" />
                    选择视频
                  </Button>
                  <p className="mt-4 text-xs text-white/35">MP4 · MOV · M4V · WEBM · 导入后自动开始本地识别</p>
                </div>
                {dragging && (
                  <div className="absolute inset-5 grid place-items-center rounded-3xl border border-dashed border-[#efbd63] bg-[#163f35]/90 text-sm font-semibold text-[#ffdf9e] backdrop-blur">
                    松开即可导入视频
                  </div>
                )}
              </div>
            )}
            <input ref={fileInputRef} accept="video/mp4,video/quicktime,video/webm,.m4v" className="sr-only" onChange={onFileChange} type="file" />

          </div>

          <div className="border-t border-white/8 bg-[#10231f] px-4 pb-4 pt-3 text-white sm:px-5">
            <div className="mb-3">
              <input
                aria-label="视频播放进度"
                className="video-progress w-full"
                disabled={!videoUrl || !duration}
                max={duration || 0}
                min="0"
                onChange={(event) => seekVideo(event.target.value)}
                step="0.05"
                style={{ '--seek-progress': `${progressPercent}%` } as CSSProperties}
                type="range"
                value={Math.min(currentTime, duration || 0)}
              />
              <div className="mt-1.5 flex items-center justify-between font-mono text-[11px] tabular-nums text-white/55">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration || captions.at(-1)?.end || 0)}</span>
              </div>
            </div>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Button aria-label="上一句" className="rounded-full border-white/12 bg-white/6 text-white hover:bg-white/12" disabled={activeIndex === 0} onClick={() => seekTo(activeIndex - 1)} size="icon-lg" variant="outline">
                  <ChevronLeft />
                </Button>
                <Button aria-label={isPlaying ? '暂停' : '播放'} className="size-11 rounded-full bg-[#efbd63] text-[#1d2824] hover:bg-[#f5cb7d]" onClick={togglePlay} size="icon-lg">
                  {isPlaying ? <Pause fill="currentColor" /> : <Play className="translate-x-px" fill="currentColor" />}
                </Button>
                <Button aria-label="下一句" className="rounded-full border-white/12 bg-white/6 text-white hover:bg-white/12" disabled={activeIndex === captions.length - 1} onClick={() => seekTo(activeIndex + 1)} size="icon-lg" variant="outline">
                  <ChevronRight />
                </Button>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  aria-pressed={loopSentence}
                  className={`rounded-full border-white/12 text-white hover:bg-white/12 ${loopSentence ? 'bg-[#356d5d]' : 'bg-white/6'}`}
                  onClick={() => setLoopSentence((value) => !value)}
                  size="sm"
                  variant="outline"
                >
                  <Repeat2 data-icon="inline-start" />
                  <span className="hidden sm:inline">单句循环</span>
                </Button>
                <NativeSelect aria-label="播放速度" className="min-w-[86px] text-white" onChange={(event) => setPlaybackRate(event.target.value)} size="sm" value={String(playbackRate)}>
                  <NativeSelectOption value="0.5">0.5×</NativeSelectOption>
                  <NativeSelectOption value="0.75">0.75×</NativeSelectOption>
                  <NativeSelectOption value="1">1.0×</NativeSelectOption>
                  <NativeSelectOption value="1.25">1.25×</NativeSelectOption>
                  <NativeSelectOption value="1.5">1.5×</NativeSelectOption>
                  <NativeSelectOption value="2">2.0×</NativeSelectOption>
                </NativeSelect>
              </div>
            </div>
          </div>
        </section>

        <aside className="flex min-h-[620px] flex-col overflow-hidden rounded-[26px] border border-black/6 bg-[#fffdf8] shadow-[0_24px_70px_rgba(20,41,35,.08)] lg:h-[calc(100vh-98px)] lg:min-h-[630px]">
          <div className="border-b border-black/6 px-4 pb-0 pt-4 sm:px-6 sm:pt-5">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-[10px] font-bold uppercase tracking-[.18em] text-[#a27757]">{hasOwnLesson ? 'Your lesson' : 'Preview lesson'}</p>
                <h2 className="mt-1 truncate text-lg font-semibold tracking-[-0.025em]">
                  {file ? file.name.replace(/\.[^.]+$/, '') : 'Daily English · Episode 01'}
                </h2>
                {file && <p className="mt-1 text-xs text-[#88908c]">{fileSize(file.size)} · {captions.length} 句</p>}
              </div>
              {file ? (
                <Button className="h-9 shrink-0 rounded-full bg-[#173f35] px-4 text-[#f7f3e9] hover:bg-[#245448]" disabled={processing} onClick={() => void generateLesson()}>
                  {processing ? <LoaderCircle className="animate-spin" data-icon="inline-start" /> : <WandSparkles data-icon="inline-start" />}
                  {processing ? '自动识别中' : '重新识别'}
                </Button>
              ) : (
                <span className="flex shrink-0 items-center gap-1.5 rounded-full bg-[#e5f2eb] px-3 py-1.5 text-xs font-semibold text-[#2d7059]"><Check size={13} /> 演示</span>
              )}
            </div>

            {processing && (
              <Progress className="mb-4" value={progress}>
                <ProgressLabel className="text-xs text-[#66706c]">本地模型流式识别中</ProgressLabel>
                <span className="ml-auto text-xs tabular-nums text-[#7b837f]">{Math.round(progress)}%</span>
              </Progress>
            )}

            <nav aria-label="字幕显示模式" className="grid grid-cols-4 text-center text-sm text-[#8b918e]">
              {MODE_OPTIONS.map((item) => (
                <button
                  key={item.value}
                  className={`relative pb-3.5 transition-colors ${mode === item.value ? 'font-semibold text-[#183f35]' : 'hover:text-[#4d5753]'}`}
                  onClick={() => setMode(item.value)}
                  title={item.note}
                >
                  {item.label}
                  {mode === item.value && <span className="absolute inset-x-3 bottom-0 h-0.5 rounded-full bg-[#e69d4f]" />}
                </button>
              ))}
            </nav>
          </div>

          <div className="flex items-center justify-between border-b border-black/5 bg-[#fbf8f1] px-4 py-2.5 text-xs sm:px-6">
            <p className="flex min-w-0 items-center gap-2 text-[#66706c]">
              {error ? <CircleHelp className="shrink-0 text-[#bd6548]" size={14} /> : <Volume2 className="shrink-0" size={14} />}
              <span className={`truncate ${error ? 'text-[#a74d34]' : ''}`}>{error || notice}</span>
            </p>
            <span className="ml-3 shrink-0 rounded-full bg-white px-2 py-1 font-mono text-[10px] text-[#8b918e]">{captions.length ? activeIndex + 1 : 0}/{captions.length}</span>
          </div>

          <div className="transcript-scroll flex-1 overflow-y-auto p-3 sm:p-4">
            <div className="space-y-2.5">
              {!captions.length && (
                <div className="grid min-h-64 place-items-center rounded-2xl border border-dashed border-[#b9c9c2] bg-[#f8faf7] px-8 text-center">
                  <div>
                    {processing ? <LoaderCircle className="mx-auto mb-4 animate-spin text-[#33705d]" size={26} /> : <Volume2 className="mx-auto mb-4 text-[#71847d]" size={26} />}
                    <p className="text-sm font-semibold text-[#40514a]">{processing ? '字幕会在识别到后逐句出现在这里' : '等待识别结果'}</p>
                    <p className="mt-2 text-xs leading-5 text-[#88938e]">不会再显示与当前视频无关的演示字幕</p>
                  </div>
                </div>
              )}
              {captions.map((caption, captionIndex) => {
                const active = captionIndex === activeIndex;
                return (
                  <article
                    key={`${caption.id}-${caption.start}`}
                    ref={(node) => { captionRefs.current[captionIndex] = node; }}
                    className={`group cursor-pointer rounded-2xl border p-4 transition-all sm:p-5 ${
                      active
                        ? 'border-[#d5a95e]/45 bg-[#fff1c9] shadow-[0_8px_28px_rgba(123,86,35,.09)]'
                        : 'border-black/6 bg-white hover:border-[#c9d7d1] hover:bg-[#fcfdfb]'
                    }`}
                    onClick={() => seekTo(captionIndex, false)}
                  >
                    <div className={`mb-3 flex items-center justify-between font-mono text-[11px] ${active ? 'text-[#8b6f4a]' : 'text-[#969c99]'}`}>
                      <span>{formatTime(caption.start)} — {formatTime(caption.end)}</span>
                      <span>{active ? 'NOW' : String(captionIndex + 1).padStart(2, '0')}</span>
                    </div>

                    {mode !== 'chinese' && (
                      <p className="text-[18px] font-semibold leading-[1.72] tracking-[-0.012em] sm:text-[20px]">
                        {caption.words.length ? caption.words.map((word, wordIndex) => {
                          const isBlank = mode === 'cloze' && (word.highlight !== 'none' || wordIndex % 5 === 2);
                          const isSpeaking = active && currentTime >= word.start && currentTime < word.end;
                          return (
                            <button
                              key={`${word.text}-${wordIndex}`}
                              aria-label={isBlank ? `显示单词 ${word.text}` : `切换 ${word.text} 的高亮颜色`}
                              className={`${HIGHLIGHT_CLASSES[word.highlight]} word-button mr-[.2em] inline rounded-[.28em] text-left transition-all focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#3c7b67] ${isBlank ? 'word-blank' : ''} ${isSpeaking ? 'word-speaking' : ''}`}
                              onClick={(event) => { event.stopPropagation(); cycleHighlight(captionIndex, wordIndex); }}
                              type="button"
                            >
                              {word.text}
                            </button>
                          );
                        }) : (caption.english || (processing ? '英文翻译生成中…' : '暂无英文翻译'))}
                      </p>
                    )}

                    {mode !== 'english' && mode !== 'cloze' && (
                      <p className={`${mode === 'chinese' ? 'text-[17px] font-medium text-[#333b37]' : 'mt-2.5 text-[14px] text-[#6f7169]'} leading-7`}>
                        {caption.sourceLanguage === 'zh' && caption.sourceWords?.length
                          ? caption.sourceWords.map((word, wordIndex) => {
                              const isSpeaking = active && currentTime >= word.start && currentTime < word.end;
                              return <span key={`${word.text}-${wordIndex}`} className={`source-word ${isSpeaking ? 'word-speaking' : ''}`}>{word.text}</span>;
                            })
                          : caption.chinese || (processing ? '中文翻译生成中…' : '暂无中文翻译')}
                      </p>
                    )}

                    {active && mode !== 'chinese' && (
                      <div className="mt-3 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[.1em] text-[#917550] opacity-70">
                        <Highlighter size={12} /> 点击单词切换高亮
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-black/6 bg-white/70 px-4 py-3 sm:px-6">
            <Button disabled={activeIndex === 0} onClick={() => seekTo(activeIndex - 1)} variant="ghost">
              <ChevronLeft data-icon="inline-start" /> 上一句
            </Button>
            <Button aria-label="重播当前句" className="rounded-full bg-[#173f35] text-white hover:bg-[#245448]" onClick={() => seekTo(activeIndex)} size="icon-lg">
              <RotateCcw />
            </Button>
            <Button disabled={activeIndex === captions.length - 1} onClick={() => seekTo(activeIndex + 1)} variant="ghost">
              下一句 <ChevronRight data-icon="inline-end" />
            </Button>
          </div>
        </aside>
      </section>
    </main>
  );
}
