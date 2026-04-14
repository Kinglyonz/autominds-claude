/**
 * Groq Fetch Adapter
 *
 * Intercepts fetch calls from the Anthropic SDK and routes them to
 * Groq's OpenAI-compatible API, translating between Anthropic Messages
 * format and OpenAI Chat Completions format.
 *
 * Supports:
 * - Text messages (user/assistant/system)
 * - Tool definitions (Anthropic input_schema → OpenAI parameters)
 * - Tool use (tool_use → tool_calls, tool_result → tool role messages)
 * - Streaming events translation
 *
 * Endpoint: https://api.groq.com/openai/v1/chat/completions
 */

// ── Available Groq models ───────────────────────────────────────────
export const GROQ_MODELS = [
  { id: 'llama-3.3-70b-versatile', label: 'Llama 3.3 70B', description: 'Fast 70B model' },
  { id: 'llama-3.1-8b-instant', label: 'Llama 3.1 8B', description: 'Ultra-fast small model' },
  { id: 'llama3-70b-8192', label: 'Llama 3 70B', description: 'Llama 3 70B' },
  { id: 'mixtral-8x7b-32768', label: 'Mixtral 8x7B', description: '32K context MoE' },
  { id: 'gemma2-9b-it', label: 'Gemma 2 9B', description: 'Google Gemma 2' },
  { id: 'deepseek-r1-distill-llama-70b', label: 'DeepSeek R1 70B', description: 'Reasoning model' },
  { id: 'qwen-qwq-32b', label: 'Qwen QWQ 32B', description: 'Qwen reasoning model' },
] as const

export const DEFAULT_GROQ_MODEL = 'llama-3.3-70b-versatile'

export function mapClaudeModelToGroq(claudeModel: string | null): string {
  if (!claudeModel) return DEFAULT_GROQ_MODEL
  if (isGroqModel(claudeModel)) return claudeModel
  const lower = claudeModel.toLowerCase()
  if (lower.includes('opus')) return 'deepseek-r1-distill-llama-70b'
  if (lower.includes('haiku')) return 'llama-3.1-8b-instant'
  if (lower.includes('sonnet')) return 'llama-3.3-70b-versatile'
  return DEFAULT_GROQ_MODEL
}

export function isGroqModel(model: string): boolean {
  return GROQ_MODELS.some(m => m.id === model)
}

// ── Types ───────────────────────────────────────────────────────────

interface AnthropicContentBlock {
  type: string
  text?: string
  id?: string
  name?: string
  input?: Record<string, unknown>
  tool_use_id?: string
  content?: string | AnthropicContentBlock[]
  source?: { type?: string; media_type?: string; data?: string }
  [key: string]: unknown
}

interface AnthropicMessage {
  role: string
  content: string | AnthropicContentBlock[]
}

interface AnthropicTool {
  name: string
  description?: string
  input_schema?: Record<string, unknown>
}

interface OpenAIMessage {
  role: string
  content?: string | null
  tool_calls?: Array<{
    id: string
    type: 'function'
    function: { name: string; arguments: string }
  }>
  tool_call_id?: string
  name?: string
}

// ── Tool translation ────────────────────────────────────────────────

function translateTools(anthropicTools: AnthropicTool[]): Array<Record<string, unknown>> {
  return anthropicTools.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description || '',
      parameters: tool.input_schema || { type: 'object', properties: {} },
    },
  }))
}

// ── Message translation: Anthropic → OpenAI ─────────────────────────

function translateMessages(
  anthropicMessages: AnthropicMessage[],
  systemPrompt: string,
): OpenAIMessage[] {
  const messages: OpenAIMessage[] = []

  // System prompt first
  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt })
  }

  for (const msg of anthropicMessages) {
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role, content: msg.content })
      continue
    }

    if (!Array.isArray(msg.content)) continue

    if (msg.role === 'user') {
      // Collect text blocks and tool_result blocks
      const textParts: string[] = []
      for (const block of msg.content) {
        if (block.type === 'tool_result') {
          let outputText = ''
          if (typeof block.content === 'string') {
            outputText = block.content
          } else if (Array.isArray(block.content)) {
            outputText = block.content
              .filter(c => c.type === 'text')
              .map(c => c.text || '')
              .join('\n')
          }
          messages.push({
            role: 'tool',
            tool_call_id: block.tool_use_id || `call_${Date.now()}`,
            content: outputText || '(empty)',
          })
        } else if (block.type === 'text' && typeof block.text === 'string') {
          textParts.push(block.text)
        }
      }
      if (textParts.length > 0) {
        messages.push({ role: 'user', content: textParts.join('\n') })
      }
    } else if (msg.role === 'assistant') {
      // Collect text and tool_use blocks
      const textParts: string[] = []
      const toolCalls: Array<{
        id: string
        type: 'function'
        function: { name: string; arguments: string }
      }> = []

      for (const block of msg.content) {
        if (block.type === 'text' && typeof block.text === 'string') {
          textParts.push(block.text)
        } else if (block.type === 'tool_use') {
          toolCalls.push({
            id: block.id || `call_${Date.now()}`,
            type: 'function',
            function: {
              name: block.name || '',
              arguments: JSON.stringify(block.input || {}),
            },
          })
        }
        // Skip 'thinking' blocks — Groq doesn't support them
      }

      const assistantMsg: OpenAIMessage = {
        role: 'assistant',
        content: textParts.length > 0 ? textParts.join('\n') : null,
      }
      if (toolCalls.length > 0) {
        assistantMsg.tool_calls = toolCalls
      }
      messages.push(assistantMsg)
    }
  }

  return messages
}

// ── SSE Formatting ──────────────────────────────────────────────────

function formatSSE(event: string, data: string): string {
  return `event: ${event}\ndata: ${data}\n\n`
}

// ── Response translation: OpenAI SSE → Anthropic SSE ────────────────

async function translateGroqStreamToAnthropic(
  groqResponse: Response,
  groqModel: string,
): Promise<Response> {
  const messageId = `msg_groq_${Date.now()}`

  const readable = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      let contentBlockIndex = 0
      let outputTokens = 0
      let inputTokens = 0
      let currentTextBlockStarted = false
      let inToolCall = false
      let hadToolCalls = false
      let currentToolCallId = ''
      let currentToolCallName = ''
      let currentToolCallArgs = ''
      // Track active tool calls by index
      const activeToolCalls: Map<number, { id: string; name: string; args: string }> = new Map()

      // Emit Anthropic message_start
      controller.enqueue(
        encoder.encode(
          formatSSE(
            'message_start',
            JSON.stringify({
              type: 'message_start',
              message: {
                id: messageId,
                type: 'message',
                role: 'assistant',
                content: [],
                model: groqModel,
                stop_reason: null,
                stop_sequence: null,
                usage: { input_tokens: 0, output_tokens: 0 },
              },
            }),
          ),
        ),
      )

      // Emit ping
      controller.enqueue(
        encoder.encode(formatSSE('ping', JSON.stringify({ type: 'ping' }))),
      )

      try {
        const reader = groqResponse.body?.getReader()
        if (!reader) {
          emitTextBlock(controller, encoder, contentBlockIndex, 'Error: No response body')
          finishStream(controller, encoder, outputTokens, inputTokens, false)
          return
        }

        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed || trimmed.startsWith(':')) continue
            if (!trimmed.startsWith('data: ')) continue
            const dataStr = trimmed.slice(6)
            if (dataStr === '[DONE]') continue

            let chunk: Record<string, unknown>
            try {
              chunk = JSON.parse(dataStr)
            } catch {
              continue
            }

            // Extract usage from chunk if present
            const usage = chunk.usage as Record<string, number> | undefined
            if (usage) {
              inputTokens = usage.prompt_tokens || inputTokens
              outputTokens = usage.completion_tokens || outputTokens
            }

            const choices = chunk.choices as Array<Record<string, unknown>> | undefined
            if (!choices || choices.length === 0) continue

            const choice = choices[0]
            const delta = choice.delta as Record<string, unknown> | undefined
            if (!delta) continue

            // ── Tool calls ──────────────────────────────────────
            const toolCalls = delta.tool_calls as Array<Record<string, unknown>> | undefined
            if (toolCalls) {
              for (const tc of toolCalls) {
                const tcIndex = (tc.index as number) ?? 0
                const fn = tc.function as Record<string, unknown> | undefined

                if (!activeToolCalls.has(tcIndex)) {
                  // New tool call starting
                  // Close text block if open
                  if (currentTextBlockStarted) {
                    controller.enqueue(
                      encoder.encode(
                        formatSSE('content_block_stop', JSON.stringify({
                          type: 'content_block_stop',
                          index: contentBlockIndex,
                        })),
                      ),
                    )
                    contentBlockIndex++
                    currentTextBlockStarted = false
                  }

                  const toolId = (tc.id as string) || `toolu_${Date.now()}_${tcIndex}`
                  const toolName = (fn?.name as string) || ''
                  activeToolCalls.set(tcIndex, { id: toolId, name: toolName, args: '' })
                  hadToolCalls = true

                  // Start tool_use block
                  controller.enqueue(
                    encoder.encode(
                      formatSSE('content_block_start', JSON.stringify({
                        type: 'content_block_start',
                        index: contentBlockIndex,
                        content_block: {
                          type: 'tool_use',
                          id: toolId,
                          name: toolName,
                          input: {},
                        },
                      })),
                    ),
                  )
                }

                // Accumulate arguments
                const argDelta = fn?.arguments as string | undefined
                if (argDelta) {
                  const tc_data = activeToolCalls.get(tcIndex)!
                  tc_data.args += argDelta
                  controller.enqueue(
                    encoder.encode(
                      formatSSE('content_block_delta', JSON.stringify({
                        type: 'content_block_delta',
                        index: contentBlockIndex,
                        delta: {
                          type: 'input_json_delta',
                          partial_json: argDelta,
                        },
                      })),
                    ),
                  )
                }
              }
              continue
            }

            // ── Text content ────────────────────────────────────
            const content = delta.content as string | undefined
            if (typeof content === 'string' && content.length > 0) {
              if (!currentTextBlockStarted) {
                controller.enqueue(
                  encoder.encode(
                    formatSSE('content_block_start', JSON.stringify({
                      type: 'content_block_start',
                      index: contentBlockIndex,
                      content_block: { type: 'text', text: '' },
                    })),
                  ),
                )
                currentTextBlockStarted = true
              }
              controller.enqueue(
                encoder.encode(
                  formatSSE('content_block_delta', JSON.stringify({
                    type: 'content_block_delta',
                    index: contentBlockIndex,
                    delta: { type: 'text_delta', text: content },
                  })),
                ),
              )
              outputTokens += 1
            }

            // ── Finish reason ───────────────────────────────────
            const finishReason = choice.finish_reason as string | undefined
            if (finishReason) {
              // Close any open tool call blocks
              for (const [_idx, _tcData] of activeToolCalls) {
                controller.enqueue(
                  encoder.encode(
                    formatSSE('content_block_stop', JSON.stringify({
                      type: 'content_block_stop',
                      index: contentBlockIndex,
                    })),
                  ),
                )
                contentBlockIndex++
              }
              activeToolCalls.clear()
            }
          }
        }
      } catch (err) {
        if (!currentTextBlockStarted) {
          controller.enqueue(
            encoder.encode(
              formatSSE('content_block_start', JSON.stringify({
                type: 'content_block_start',
                index: contentBlockIndex,
                content_block: { type: 'text', text: '' },
              })),
            ),
          )
          currentTextBlockStarted = true
        }
        controller.enqueue(
          encoder.encode(
            formatSSE('content_block_delta', JSON.stringify({
              type: 'content_block_delta',
              index: contentBlockIndex,
              delta: { type: 'text_delta', text: `\n\n[Groq Error: ${String(err)}]` },
            })),
          ),
        )
      }

      // Close remaining blocks
      if (currentTextBlockStarted) {
        controller.enqueue(
          encoder.encode(
            formatSSE('content_block_stop', JSON.stringify({
              type: 'content_block_stop',
              index: contentBlockIndex,
            })),
          ),
        )
      }

      finishStream(controller, encoder, outputTokens, inputTokens, hadToolCalls)
    },
  })

  function emitTextBlock(
    controller: ReadableStreamDefaultController,
    encoder: TextEncoder,
    index: number,
    text: string,
  ) {
    controller.enqueue(encoder.encode(formatSSE('content_block_start', JSON.stringify({
      type: 'content_block_start', index, content_block: { type: 'text', text: '' },
    }))))
    controller.enqueue(encoder.encode(formatSSE('content_block_delta', JSON.stringify({
      type: 'content_block_delta', index, delta: { type: 'text_delta', text },
    }))))
    controller.enqueue(encoder.encode(formatSSE('content_block_stop', JSON.stringify({
      type: 'content_block_stop', index,
    }))))
  }

  function finishStream(
    controller: ReadableStreamDefaultController,
    encoder: TextEncoder,
    outputTokens: number,
    inputTokens: number,
    hadToolCalls: boolean,
  ) {
    const stopReason = hadToolCalls ? 'tool_use' : 'end_turn'
    controller.enqueue(encoder.encode(formatSSE('message_delta', JSON.stringify({
      type: 'message_delta',
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }))))
    controller.enqueue(encoder.encode(formatSSE('message_stop', JSON.stringify({
      type: 'message_stop',
      usage: { input_tokens: inputTokens, output_tokens: outputTokens },
    }))))
    controller.close()
  }

  return new Response(readable, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'x-request-id': messageId,
    },
  })
}

// ── Main fetch interceptor ──────────────────────────────────────────

const GROQ_BASE_URL = 'https://api.groq.com/openai/v1/chat/completions'

/**
 * Creates a fetch function that intercepts Anthropic API calls and routes them to Groq.
 */
export function createGroqFetch(
  apiKey: string,
): (input: RequestInfo | URL, init?: RequestInit) => Promise<Response> {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = input instanceof Request ? input.url : String(input)

    // Only intercept Anthropic message calls
    if (!url.includes('/v1/messages')) {
      return globalThis.fetch(input, init)
    }

    // Parse the Anthropic request body
    let anthropicBody: Record<string, unknown>
    try {
      const bodyText =
        init?.body instanceof ReadableStream
          ? await new Response(init.body).text()
          : typeof init?.body === 'string'
            ? init.body
            : '{}'
      anthropicBody = JSON.parse(bodyText)
    } catch {
      anthropicBody = {}
    }

    // Extract and translate
    const anthropicMessages = (anthropicBody.messages || []) as AnthropicMessage[]
    const anthropicTools = (anthropicBody.tools || []) as AnthropicTool[]
    const claudeModel = anthropicBody.model as string
    const systemPrompt = anthropicBody.system as
      | string
      | Array<{ type: string; text?: string }>
      | undefined

    const groqModel = mapClaudeModelToGroq(claudeModel)

    // Build system instructions
    let instructions = ''
    if (systemPrompt) {
      instructions =
        typeof systemPrompt === 'string'
          ? systemPrompt
          : Array.isArray(systemPrompt)
            ? systemPrompt
                .filter(b => b.type === 'text' && typeof b.text === 'string')
                .map(b => b.text!)
                .join('\n')
            : ''
    }

    const openaiMessages = translateMessages(anthropicMessages, instructions)

    // Build Groq request
    const groqBody: Record<string, unknown> = {
      model: groqModel,
      messages: openaiMessages,
      stream: true,
      stream_options: { include_usage: true },
      max_tokens: (anthropicBody.max_tokens as number) || 8192,
    }

    if (anthropicTools.length > 0) {
      groqBody.tools = translateTools(anthropicTools)
      groqBody.tool_choice = 'auto'
    }

    // Call Groq API
    const groqResponse = await globalThis.fetch(GROQ_BASE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(groqBody),
    })

    if (!groqResponse.ok) {
      const errorText = await groqResponse.text()
      const errorBody = {
        type: 'error',
        error: {
          type: 'api_error',
          message: `Groq API error (${groqResponse.status}): ${errorText}`,
        },
      }
      return new Response(JSON.stringify(errorBody), {
        status: groqResponse.status,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    return translateGroqStreamToAnthropic(groqResponse, groqModel)
  }
}
