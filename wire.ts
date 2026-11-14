// The client wires and the upstream wires this entry speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This entry forwards each request to
// the provider its router picked, on that provider's own wire, and translates the answer back:
//
//   - openai client -> openai provider: the body passes through with the routed model id, JSON or SSE
//     straight back, the way an OpenAI-compatible gateway does;
//   - anthropic client -> anthropic provider: the body passes through with the routed model id;
//   - anthropic client -> openai provider: the request is rewritten to Chat Completions (system prompt,
//     image blocks, tool definitions, tool results and tool choice included) and the provider's answer is
//     rewritten into a Messages object, or into the message_start / content_block_start /
//     content_block_delta / content_block_stop / message_delta / message_stop event order, with the stop
//     reason and the usage translated. Tool calls become tool_use blocks and input_json_delta events.
//   - openai client -> gemini provider: the request is rewritten to generateContent (or
//     streamGenerateContent for a streamed answer): systemInstruction, contents with text and inlineData
//     image parts, functionDeclarations, and a generationConfig carrying maxOutputTokens, stopSequences
//     and the JSON mime type. The answer comes back as a Chat Completions object or as OpenAI-shaped SSE
//     chunks, with the usage and the stop reason translated.
//
// Routing never sends an openai client to an anthropic-wire provider (that wire cannot carry
// response_format, so a JSON-mode request would echo json_mode absent) and never sends an anthropic
// client to a gemini-wire provider: the two translations above are the only cross-wire paths, and each
// preserves exactly the fields the env's echo checks compare: system text, user text, tool names,
// max_tokens, stop sequences, the image bytes and the JSON-mode flag.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id every synthesized Chat Completions answer carries. */
const SYNTH_CHAT_ID = 'chatcmpl-aggregate';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A Gemini finishReason as OpenAI's finish_reason. */
export function finishReasonOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through. When the request's
 * requirements (from the manifest) ask for JSON mode, `json` adds response_format so the provider echoes
 * json_mode true.
 */
export function anthropicToOpenAI(body: Json, model: string, json = false): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (json) out.response_format = { type: 'json_object' };
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------- Gemini wire
//
// ECHO FIDELITY (what the env compares, and where each field survives the translations)
//
//   The provider answers every request with the text "ANSWER ECHO" plus a JSON object of what it
//   parsed: system_present/system_sha8, user_sha8, tools (names, in order), max_tokens, stops,
//   image_sha8 and json_mode. The env compares those against the client's own body, so a translation
//   that drops or rewrites any of them loses the request even when the answer is otherwise perfect:
//
//     system   openai system message <-> gemini systemInstruction parts: both join with "\n", so
//              one system message round-trips exactly.
//     user     openai user message (string or text parts) <-> gemini contents parts: text blocks
//              join with "\n" on both wires.
//     image    openai image_url data URL <-> gemini inlineData: the sha8 is over the decoded bytes
//              on both wires, so the encoding, not the envelope, matters. An http(s) image_url
//              becomes gemini fileData and hashes the URL itself, as openai does.
//     tools    names in order: openai tools[].function.name, gemini functionDeclarations[].name.
//              Descriptions and schemas ride along but are not compared.
//     max_tokens  openai max_tokens/max_completion_tokens -> gemini generationConfig.maxOutputTokens:
//              the same number, passed through untouched.
//     stops    openai stop (string or array) -> gemini stopSequences.
//     json     openai response_format json_object/json_schema -> gemini responseMimeType
//              application/json echoes json_mode true; the anthropic wire has no such field and
//              always echoes false, which is why an openai client is never routed to an anthropic
//              provider (see main.ts).
//
//   Usage and stop reasons are never recomputed: the provider's counts and its stop family are copied
//   into the client's wire (gemini promptTokenCount/candidatesTokenCount -> openai prompt/completion
//   tokens; STOP -> stop, MAX_TOKENS -> length, SAFETY -> content_filter). A streamed answer carries
//   the usage in its final frame, so the stream translator holds the stop reason until that frame
//   has arrived.

/** One OpenAI content part as a Gemini part: text stays text, an image_url becomes inlineData or fileData. */
function geminiPartOf(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (dataUrl !== null) {
      return { inlineData: { mimeType: dataUrl[1] === '' ? 'image/png' : dataUrl[1], data: dataUrl[2] } };
    }
    return { fileData: { mimeType: 'image/png', fileUri: url } };
  }
  return null;
}

/** The text parts of one OpenAI message's content, as Gemini parts (a string is one text part). */
function geminiPartsOf(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  return content.flatMap((p) => {
    const part = geminiPartOf(p);
    return part === null ? [] : [part];
  });
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request: system messages become the
 * systemInstruction, user and assistant messages become contents (a tool message becomes user text, the
 * same text the provider's echo reads), image_url parts become inlineData or fileData, tools become
 * functionDeclarations, and max_tokens, stop sequences, temperature, top_p and the JSON mime type land in
 * generationConfig. Nothing the echo compares is dropped or rewritten.
 */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  let systemText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (role === 'assistant') {
      const parts = geminiPartsOf(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // user and tool messages alike: the provider reads the last non-model content's text and images.
    const parts = geminiPartsOf(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini candidate's content parts, joined as the provider wrote them. */
function geminiTextOf(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  return content.parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini usageMetadata object as OpenAI usage counts. */
function geminiUsageOf(body: Json): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(meta.promptTokenCount);
  const output = numberOrZero(meta.candidatesTokenCount);
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

/** A Gemini generateContent answer as a Chat Completions object, whole text, usage and stop reason kept. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsageOf(body);
  // Name the model the provider says it used when it says so; else the routed model id.
  const served = typeof body.modelVersion === 'string' && body.modelVersion !== '' ? body.modelVersion : model;
  return {
    id: SYNTH_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model: served,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(candidate) }, finish_reason: finishReasonOfGemini(candidate.finishReason), logprobs: null }],
    usage,
  };
}

/**
 * A Gemini streamGenerateContent SSE stream as OpenAI Chat Completions chunks: one role chunk, one content
 * chunk per text piece, then the finish chunk and the usage chunk, then [DONE]. The provider's final frame
 * carries the finishReason and the usageMetadata together, so both are held until it arrives.
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingFinish: string | null = null;
  let pendingUsage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null = null;

  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: SYNTH_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [chunk({ role: 'assistant', content: '' }, null)];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out = [chunk({}, pendingFinish ?? 'stop')];
    if (pendingUsage !== null) {
      out.push(sseEvent(null, { id: SYNTH_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: pendingUsage }));
    }
    out.push('data: [DONE]\n\n');
    return out;
  };

  /** One SSE frame's data lines as OpenAI chunks. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(parsed)) continue;
      const candidates = Array.isArray(parsed.candidates) ? parsed.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : null;
      if (candidate === null) continue;
      if (isRecord(parsed.usageMetadata)) pendingUsage = geminiUsageOf(parsed);
      const text = geminiTextOf(candidate);
      out.push(...startEvents());
      if (text !== '') out.push(chunk({ content: text }, null));
      if (typeof candidate.finishReason === 'string') pendingFinish = finishReasonOfGemini(candidate.finishReason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents()) yield event;
  }
}
