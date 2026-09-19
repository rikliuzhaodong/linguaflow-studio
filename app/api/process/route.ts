type WhisperWord = { word: string; start: number; end: number };
type WhisperSegment = { id?: number; start: number; end: number; text: string };
type WhisperResponse = { text?: string; words?: WhisperWord[]; segments?: WhisperSegment[] };
type TranslationResponse = { output?: Array<{ type?: string; content?: Array<{ type?: string; text?: string }> }> };

const MAX_FILE_SIZE = 25 * 1024 * 1024;

export async function POST(request: Request) {
  try {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return Response.json(
        { error: '尚未配置 OPENAI_API_KEY。请复制 .env.example 为 .env.local 并填入密钥。' },
        { status: 503 },
      );
    }

    const incoming = await request.formData();
    const file = incoming.get('file');
    if (!(file instanceof File)) return Response.json({ error: '没有收到视频文件。' }, { status: 400 });
    if (file.size > MAX_FILE_SIZE) {
      return Response.json({ error: '文件超过 25 MB，请压缩视频或导出音频后重试。' }, { status: 413 });
    }

    const transcriptionBody = new FormData();
    transcriptionBody.append('file', file, file.name || 'lesson.mp4');
    transcriptionBody.append('model', 'whisper-1');
    transcriptionBody.append('language', 'en');
    transcriptionBody.append('response_format', 'verbose_json');
    transcriptionBody.append('timestamp_granularities[]', 'word');

    const transcriptionRequest = await fetch('https://api.openai.com/v1/audio/transcriptions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}` },
      body: transcriptionBody,
    });
    if (!transcriptionRequest.ok) {
      const message = await safeApiMessage(transcriptionRequest);
      return Response.json({ error: `语音识别失败：${message}` }, { status: transcriptionRequest.status });
    }

    const transcript = (await transcriptionRequest.json()) as WhisperResponse;
    const segments = normalizeSegments(transcript);
    if (!segments.length) return Response.json({ error: '没有检测到清晰的英文语音。' }, { status: 422 });

    let translations: string[] = [];
    try {
      translations = await translateSegments(apiKey, segments.map((segment) => segment.text));
    } catch {
      translations = segments.map(() => '中文翻译暂不可用，可先使用英文模式学习。');
    }

    const captions = segments.map((segment, index) => {
      const words = (transcript.words ?? []).filter((word) => {
        const middle = (word.start + word.end) / 2;
        return middle >= segment.start && middle <= segment.end;
      });
      const normalizedWords = words.length ? words : synthesizeWords(segment.text, segment.start, segment.end);
      return {
        id: index + 1,
        start: segment.start,
        end: segment.end,
        english: segment.text,
        chinese: translations[index] ?? '中文翻译暂不可用。',
        words: normalizedWords.map((word, wordIndex) => ({
          text: word.word.trim(),
          start: word.start,
          end: word.end,
          highlight: chooseHighlight(word.word, wordIndex),
        })),
      };
    });

    return Response.json({ captions });
  } catch (error) {
    const message = error instanceof Error ? error.message : '未知错误';
    return Response.json({ error: `处理视频时出错：${message}` }, { status: 500 });
  }
}

function normalizeSegments(transcript: WhisperResponse): WhisperSegment[] {
  const source = transcript.segments?.filter((segment) => segment.text.trim()) ?? [];
  if (source.length) {
    return source.map((segment, index) => ({
      id: segment.id ?? index,
      start: Number(segment.start) || 0,
      end: Math.max(Number(segment.end) || 0, (Number(segment.start) || 0) + 0.5),
      text: segment.text.trim(),
    }));
  }
  const text = transcript.text?.trim();
  if (!text) return [];
  const words = transcript.words ?? [];
  return [{ id: 0, start: words[0]?.start ?? 0, end: words.at(-1)?.end ?? 5, text }];
}

function synthesizeWords(text: string, start: number, end: number): WhisperWord[] {
  const parts = text.split(/\s+/).filter(Boolean);
  const unit = (end - start) / Math.max(parts.length, 1);
  return parts.map((word, index) => ({ word, start: start + unit * index, end: start + unit * (index + 1) }));
}

function chooseHighlight(word: string, index: number): 'none' | 'mint' | 'orange' | 'blue' {
  const clean = word.replace(/[^a-z'-]/gi, '');
  if (clean.length >= 8) return 'blue';
  if (clean.length >= 6 && index % 2 === 0) return 'mint';
  if (clean.length >= 5 && index % 3 === 0) return 'orange';
  return 'none';
}

async function translateSegments(apiKey: string, lines: string[]): Promise<string[]> {
  const model = process.env.OPENAI_TEXT_MODEL || 'gpt-5-mini';
  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      store: false,
      instructions: 'Translate English learning subtitles into natural, concise Simplified Chinese. Preserve meaning, tone, names, and sentence order. Return only the requested structured data.',
      input: JSON.stringify(lines),
      text: {
        format: {
          type: 'json_schema',
          name: 'subtitle_translations',
          strict: true,
          schema: {
            type: 'object',
            properties: { translations: { type: 'array', items: { type: 'string' } } },
            required: ['translations'],
            additionalProperties: false,
          },
        },
      },
    }),
  });
  if (!response.ok) throw new Error(await safeApiMessage(response));
  const payload = (await response.json()) as TranslationResponse;
  const outputText = payload.output?.flatMap((item) => item.content ?? []).find((item) => item.type === 'output_text')?.text;
  if (!outputText) throw new Error('翻译响应为空');
  const parsed = JSON.parse(outputText) as { translations?: string[] };
  if (!Array.isArray(parsed.translations)) throw new Error('翻译响应格式不正确');
  return parsed.translations;
}

async function safeApiMessage(response: Response) {
  try {
    const body = (await response.json()) as { error?: { message?: string } };
    return body.error?.message || `请求失败（${response.status}）`;
  } catch {
    return `请求失败（${response.status}）`;
  }
}
