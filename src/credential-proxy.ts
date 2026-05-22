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
