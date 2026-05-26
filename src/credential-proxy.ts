/**
 * Credential proxy for container isolation.
 * Containers connect here instead of directly to the model API. The proxy
 * injects real credentials so containers never see them.
 *
 * In NanoClaw's current setup this proxy is used for the **Vertex AI** auth
 * mode only — OneCLI's gateway handles API key / OAuth (see container-runner).
 * The api-key / oauth branches below are kept intact so the proxy remains a
 * complete, standalone credential injector if OneCLI is ever swapped out.
 *
 * Three auth modes:
 *   API key:  Proxy injects x-api-key on every request.
 *   OAuth:    Container CLI exchanges its placeholder token for a temp
 *             API key via /api/oauth/claude_cli/create_api_key.
 *             Proxy injects real OAuth token on that exchange request;
 *             subsequent requests carry the temp key which is valid as-is.
 *   Vertex:   Container SDK sends requests here with CLAUDE_CODE_SKIP_VERTEX_AUTH=1
 *             (no GCP creds in container). Proxy obtains Google OAuth2 tokens
 *             on the host and injects Bearer auth before forwarding to the
 *             real Vertex AI endpoint.
 */
import { createServer, Server } from 'http';
import { request as httpsRequest } from 'https';
import { request as httpRequest, RequestOptions } from 'http';
import { appendFile } from 'fs/promises';
import { resolve as resolvePath } from 'path';
import { gunzipSync, inflateSync, brotliDecompressSync } from 'zlib';

import { readEnvFile } from './env.js';
import { log } from './log.js';

export type AuthMode = 'api-key' | 'oauth' | 'vertex';

/** Vertex AI config read from .env. */
export interface VertexConfig {
  region: string;
  projectId: string;
}

let cachedGoogleAuth: import('google-auth-library').GoogleAuth | null = null;

/**
 * Lazily obtain a Google OAuth2 access token using Application Default Credentials.
 * Caches the auth client across calls; the library handles token refresh internally.
 */
async function getGoogleAccessToken(credentialsPath?: string): Promise<string> {
  const { GoogleAuth } = await import('google-auth-library');
  if (!cachedGoogleAuth) {
    cachedGoogleAuth = new GoogleAuth({
      scopes: ['https://www.googleapis.com/auth/cloud-platform'],
      ...(credentialsPath ? { keyFile: credentialsPath } : {}),
    });
  }
  const token = await cachedGoogleAuth.getAccessToken();
  if (!token) throw new Error('Failed to obtain Google access token');
  return token;
}

// Keywords Gemini's function-declaration schema rejects. Vertex's OpenAI-compatible
// endpoint validates tool parameter schemas against a restricted OpenAPI subset and
// returns INVALID_ARGUMENT on these. $ref/ref/$defs are dropped (Gemini has no schema
// references); oneOf is rewritten to anyOf (which Gemini does support).
const GEMINI_UNSUPPORTED_SCHEMA_KEYS = new Set([
  '$schema',
  '$id',
  '$ref',
  'ref',
  '$defs',
  'definitions',
  'additionalProperties',
  'allOf',
  'not',
  'const',
  'examples',
  'default',
  'format',
  'pattern',
  'multipleOf',
  'uniqueItems',
  'minLength',
  'maxLength',
  'minimum',
  'maximum',
  'title',
]);

// Schema-aware prune. Must distinguish JSON-Schema *keywords* (which may be
// unsupported) from *property names* (arbitrary, e.g. a tool with a property
// literally named "pattern" or "format") — so we only strip keywords at the
// schema level and recurse into subschema-bearing keywords, never into property
// names. (Treating property names as keywords was deleting real properties and
// leaving dangling `required` entries, which Gemini also rejects.)
function pruneSchemaForGemini(schema: unknown): void {
  if (Array.isArray(schema)) {
    for (const item of schema) pruneSchemaForGemini(item);
    return;
  }
  if (!schema || typeof schema !== 'object') return;
  const obj = schema as Record<string, unknown>;

  if ('oneOf' in obj && !('anyOf' in obj)) {
    obj.anyOf = obj.oneOf;
    delete obj.oneOf;
  }
  for (const key of GEMINI_UNSUPPORTED_SCHEMA_KEYS) {
    if (key in obj) delete obj[key];
  }

  // Recurse only into keywords whose values are subschemas.
  const props = obj.properties;
  if (props && typeof props === 'object') {
    for (const sub of Object.values(props as Record<string, unknown>)) pruneSchemaForGemini(sub);
  }
  if (obj.items) pruneSchemaForGemini(obj.items);
  for (const kw of ['anyOf', 'prefixItems'] as const) {
    if (Array.isArray(obj[kw])) for (const sub of obj[kw] as unknown[]) pruneSchemaForGemini(sub);
  }

  // Safety net: drop `required` entries that have no matching property, which
  // Gemini rejects (can arise from upstream schemas or earlier transforms).
  if (Array.isArray(obj.required) && props && typeof props === 'object') {
    const known = new Set(Object.keys(props as Record<string, unknown>));
    obj.required = (obj.required as unknown[]).filter((r) => typeof r === 'string' && known.has(r));
  }
}

// ── Usage capture ────────────────────────────────────────────────────────────
// Observe Vertex chat/completions responses, extract { usage: {...} }, and
// append a JSONL record to PROXY_USAGE_LOG (default ./data/proxy-usage.jsonl).
// The deltawave pipeline reads these records (filtered by timestamp window)
// to attribute tokens + cost to a specific codegen run. Capture is fire-and-
// forget — failures log a warning but never affect the proxy response, which
// is already flowing back to the container by the time we look at usage.

const USAGE_LOG_PATH = resolvePath(process.env.PROXY_USAGE_LOG || './data/proxy-usage.jsonl');

interface UsageRecord {
  ts: string;
  url: string;
  model: string | null;
  prompt_tokens: number;
  completion_tokens: number;
}

function extractUsage(
  body: Buffer,
): { usage: { prompt_tokens?: number; completion_tokens?: number }; model: string | null } | null {
  const text = body.toString('utf8');
  if (!text) return null;
  // SSE streaming: usage rides in the last data: chunk before [DONE] when the
  // request set stream_options.include_usage. Scan backwards for the last
  // parseable JSON event carrying usage.
  if (text.includes('\ndata:') || text.startsWith('data:')) {
    const lines = text.split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === '[DONE]') continue;
      try {
        const obj = JSON.parse(payload) as {
          usage?: { prompt_tokens?: number; completion_tokens?: number };
          model?: string;
        };
        if (obj.usage) return { usage: obj.usage, model: obj.model ?? null };
      } catch {
        /* skip malformed events */
      }
    }
    return null;
  }
  // Non-streaming JSON body.
  try {
    const obj = JSON.parse(text) as {
      usage?: { prompt_tokens?: number; completion_tokens?: number };
      model?: string;
    };
    if (obj.usage) return { usage: obj.usage, model: obj.model ?? null };
  } catch {
    /* not JSON — error pages, non-OpenAI shapes, etc. */
  }
  return null;
}

function decompressIfNeeded(body: Buffer, contentEncoding?: string): Buffer {
  // Vertex AI returns gzipped responses by default for chat/completions; the
  // container's AI SDK accepts them transparently, but our observer sees the
  // raw bytes. Decompress so extractUsage can parse the JSON/SSE plaintext.
  const enc = (contentEncoding || '').toLowerCase().trim();
  if (!enc || enc === 'identity') return body;
  try {
    if (enc === 'gzip') return gunzipSync(body);
    if (enc === 'deflate') return inflateSync(body);
    if (enc === 'br') return brotliDecompressSync(body);
  } catch (err) {
    log.warn('Usage capture: decompression failed', { err, encoding: enc });
    return Buffer.alloc(0);
  }
  return body;
}

async function captureUsage(body: Buffer, url: string, contentEncoding?: string): Promise<void> {
  if (!url.includes('/chat/completions')) return;
  const decoded = decompressIfNeeded(body, contentEncoding);
  const extracted = extractUsage(decoded);
  if (!extracted) return;
  const rec: UsageRecord = {
    ts: new Date().toISOString(),
    url,
    model: extracted.model,
    prompt_tokens: extracted.usage.prompt_tokens ?? 0,
    completion_tokens: extracted.usage.completion_tokens ?? 0,
  };
  try {
    await appendFile(USAGE_LOG_PATH, JSON.stringify(rec) + '\n', 'utf8');
  } catch (err) {
    log.warn('Failed to write proxy-usage log', { err, path: USAGE_LOG_PATH });
  }
}

/**
 * Sanitize OpenAI chat/completions tool parameter schemas so Vertex's Gemini
 * OpenAI-compatible endpoint accepts them. OpenCode (via the AI SDK) emits rich
 * JSON-Schema (e.g. $schema, additionalProperties, oneOf, format) that Gemini
 * rejects. Returns the body unchanged if it isn't JSON or carries no tools, so
 * non-tool requests pass through untouched.
 */
function sanitizeGeminiToolSchemas(body: Buffer): Buffer {
  let parsed: { tools?: Array<{ function?: { parameters?: unknown } }> };
  try {
    parsed = JSON.parse(body.toString('utf8'));
  } catch {
    return body;
  }
  if (!parsed || !Array.isArray(parsed.tools)) return body;
  for (const tool of parsed.tools) {
    if (tool?.function?.parameters) pruneSchemaForGemini(tool.function.parameters);
  }
  return Buffer.from(JSON.stringify(parsed), 'utf8');
}

export function startCredentialProxy(port: number, host = '127.0.0.1'): Promise<Server> {
  const secrets = readEnvFile([
    'ANTHROPIC_API_KEY',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_BASE_URL',
    'CLAUDE_CODE_USE_VERTEX',
    'CLOUD_ML_REGION',
    'ANTHROPIC_VERTEX_PROJECT_ID',
    'GOOGLE_APPLICATION_CREDENTIALS',
  ]);

  const authMode: AuthMode = secrets.CLAUDE_CODE_USE_VERTEX
    ? 'vertex'
    : secrets.ANTHROPIC_API_KEY
      ? 'api-key'
      : 'oauth';

  const oauthToken = secrets.CLAUDE_CODE_OAUTH_TOKEN || secrets.ANTHROPIC_AUTH_TOKEN;

  const vertexRegion = secrets.CLOUD_ML_REGION;
  const upstreamUrl = new URL(
    authMode === 'vertex'
      ? vertexRegion === 'global'
        ? 'https://aiplatform.googleapis.com/v1'
        : `https://${vertexRegion}-aiplatform.googleapis.com/v1`
      : secrets.ANTHROPIC_BASE_URL || 'https://api.anthropic.com',
  );
  const isHttps = upstreamUrl.protocol === 'https:';
  const makeRequest = isHttps ? httpsRequest : httpRequest;

  const gcpCredentialsPath = secrets.GOOGLE_APPLICATION_CREDENTIALS;

  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', async () => {
        const body = Buffer.concat(chunks);
        // Vertex's Gemini OpenAI-compatible endpoint rejects unsupported tool-schema
        // keywords; sanitize chat/completions bodies in Vertex mode. Other requests
        // (Claude path, non-tool calls) pass through unchanged.
        const fwdBody =
          authMode === 'vertex' && (req.url || '').includes('/chat/completions')
            ? sanitizeGeminiToolSchemas(body)
            : body;
        const headers: Record<string, string | number | string[] | undefined> = {
          ...(req.headers as Record<string, string>),
          host: upstreamUrl.host,
          'content-length': fwdBody.length,
        };

        // Strip hop-by-hop headers that must not be forwarded by proxies
        delete headers['connection'];
        delete headers['keep-alive'];
        delete headers['transfer-encoding'];

        if (authMode === 'api-key') {
          delete headers['x-api-key'];
          headers['x-api-key'] = secrets.ANTHROPIC_API_KEY;
        } else if (authMode === 'vertex') {
          try {
            const accessToken = await getGoogleAccessToken(gcpCredentialsPath);
            delete headers['authorization'];
            headers['authorization'] = `Bearer ${accessToken}`;
          } catch (err) {
            log.error('Failed to obtain Google access token', { err });
            if (!res.headersSent) {
              res.writeHead(502);
              res.end('Failed to obtain Google credentials');
            }
            return;
          }
        } else {
          // OAuth mode: replace placeholder Bearer token with the real one
          // only when the container actually sends an Authorization header.
          if (headers['authorization']) {
            delete headers['authorization'];
            if (oauthToken) {
              headers['authorization'] = `Bearer ${oauthToken}`;
            }
          }
        }

        // For Vertex AI, the upstream URL includes a path prefix (/v1) that
        // must be prepended to the request path from the container SDK.
        const upstreamPathPrefix = upstreamUrl.pathname.replace(/\/$/, '');
        const forwardPath = upstreamPathPrefix !== '/' ? upstreamPathPrefix + req.url : req.url;

        const upstream = makeRequest(
          {
            hostname: upstreamUrl.hostname,
            port: upstreamUrl.port || (isHttps ? 443 : 80),
            path: forwardPath,
            method: req.method,
            headers,
          } as RequestOptions,
          (upRes) => {
            res.writeHead(upRes.statusCode!, upRes.headers);
            // Tee the response: pipe straight to the container AND accumulate
            // chunks so we can extract usage after end. Listeners are attached
            // before pipe() so no chunks are missed when flowing mode starts.
            const respChunks: Buffer[] = [];
            const reqUrl = req.url || '';
            upRes.on('data', (chunk: Buffer) => respChunks.push(chunk));
            const upContentEncoding = upRes.headers['content-encoding'] as string | undefined;
            upRes.on('end', () => {
              void captureUsage(Buffer.concat(respChunks), reqUrl, upContentEncoding);
            });
            upRes.pipe(res);
          },
        );

        upstream.on('error', (err) => {
          log.error('Credential proxy upstream error', { err, url: req.url });
          if (!res.headersSent) {
            res.writeHead(502);
            res.end('Bad Gateway');
          }
        });

        upstream.write(fwdBody);
        upstream.end();
      });
    });

    server.listen(port, host, () => {
      log.info('Credential proxy started', { port, host, authMode });
      resolve(server);
    });

    server.on('error', reject);
  });
}

/** Detect which auth mode the host is configured for. */
export function detectAuthMode(): AuthMode {
  const secrets = readEnvFile(['ANTHROPIC_API_KEY', 'CLAUDE_CODE_USE_VERTEX']);
  if (secrets.CLAUDE_CODE_USE_VERTEX) return 'vertex';
  return secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
}

/** Read Vertex AI config from .env (region + project ID). */
export function readVertexConfig(): VertexConfig | null {
  const env = readEnvFile(['CLOUD_ML_REGION', 'ANTHROPIC_VERTEX_PROJECT_ID']);
  if (!env.CLOUD_ML_REGION || !env.ANTHROPIC_VERTEX_PROJECT_ID) return null;
  return {
    region: env.CLOUD_ML_REGION,
    projectId: env.ANTHROPIC_VERTEX_PROJECT_ID,
  };
}
