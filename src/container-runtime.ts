/**
 * Container runtime abstraction for NanoClaw.
 * All runtime-specific logic lives here so swapping runtimes means changing one file.
 */
import { execSync } from 'child_process';
import fs from 'fs';
import os from 'os';

import { CONTAINER_INSTALL_LABEL } from './config.js';
import { log } from './log.js';

/** The container runtime binary name. */
export const CONTAINER_RUNTIME_BIN = 'docker';

/** Hostname containers use to reach the host machine. */
export const CONTAINER_HOST_GATEWAY = 'host.docker.internal';

/**
 * Address the credential proxy binds to. Containers always reach it via
 * `host.docker.internal` (see {@link hostGatewayArgs}); this controls which
 * host-side interface accepts those connections.
 *
 * Docker Desktop (macOS) / WSL2: 127.0.0.1 — the VM routes
 *   host.docker.internal to loopback.
 * Bare-metal Linux (e.g. the GX10 deployment target): bind to the docker0
 *   bridge IP so only containers can reach it. A proxy on 127.0.0.1 would be
 *   unreachable here, since host.docker.internal resolves to the bridge
 *   gateway IP, not loopback. Falls back to 0.0.0.0 if docker0 isn't found.
 *
 * Override with CREDENTIAL_PROXY_HOST (e.g. rootless Docker, custom bridge).
 */
export const PROXY_BIND_HOST = process.env.CREDENTIAL_PROXY_HOST || detectProxyBindHost();

function detectProxyBindHost(): string {
  if (os.platform() === 'darwin') return '127.0.0.1';

  // Native Docker has a docker0 bridge netdev. It's the reliable signal and
  // takes precedence over WSL detection: native docker-ce inside WSL2 has a
  // docker0 bridge and behaves like bare-metal Linux (the GX10 target), NOT
  // like Docker Desktop. Containers reach the host via host.docker.internal →
  // the bridge gateway (see hostGatewayArgs's --add-host=host-gateway), so the
  // proxy must bind to the bridge IP — a loopback bind would be unreachable.
  //
  // Use /sys (the netdev persists even when the bridge is DOWN, i.e. when no
  // containers are attached) plus `ip addr` for the address. os.networkInterfaces()
  // is NOT reliable here — it omits DOWN interfaces, and docker0 is down at host
  // startup before the first container spawns.
  if (fs.existsSync('/sys/class/net/docker0')) {
    const ip = bridgeIpV4('docker0');
    if (ip) return ip;
    // Bridge exists but its IP couldn't be read — bind all interfaces so
    // containers can still reach the proxy via the gateway.
    return '0.0.0.0';
  }

  // No docker0 netdev — Docker Desktop (macOS/Windows/WSL2) runs the engine in
  // its own VM and routes host.docker.internal to loopback. Check /proc, not
  // env vars — WSL_DISTRO_NAME isn't set under systemd.
  if (os.platform() === 'win32' || fs.existsSync('/proc/sys/fs/binfmt_misc/WSLInterop')) {
    return '127.0.0.1';
  }
  return '0.0.0.0';
}

/**
 * IPv4 address of a bridge interface, read in a way that works even when the
 * interface is DOWN. `os.networkInterfaces()` only reports UP interfaces, so
 * prefer `ip addr` (reports the configured address regardless of state) and
 * fall back to os.networkInterfaces() if the `ip` tool isn't present.
 */
function bridgeIpV4(iface: string): string | null {
  try {
    const out = execSync(`ip -4 -o addr show ${iface}`, { stdio: ['ignore', 'pipe', 'ignore'] }).toString();
    const m = out.match(/inet (\d+\.\d+\.\d+\.\d+)\//);
    if (m) return m[1];
  } catch {
    /* `ip` not available — fall through */
  }
  const entries = os.networkInterfaces()[iface];
  return entries?.find((a) => a.family === 'IPv4')?.address ?? null;
}

/** CLI args needed for the container to resolve the host gateway. */
export function hostGatewayArgs(): string[] {
  // On Linux, host.docker.internal isn't built-in — add it explicitly
  if (os.platform() === 'linux') {
    return ['--add-host=host.docker.internal:host-gateway'];
  }
  return [];
}

/** Returns CLI args for a readonly bind mount. */
export function readonlyMountArgs(hostPath: string, containerPath: string): string[] {
  return ['-v', `${hostPath}:${containerPath}:ro`];
}

/** Stop a container by name. Uses execFileSync to avoid shell injection. */
export function stopContainer(name: string): void {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(name)) {
    throw new Error(`Invalid container name: ${name}`);
  }
  execSync(`${CONTAINER_RUNTIME_BIN} stop -t 1 ${name}`, { stdio: 'pipe' });
}

/** Ensure the container runtime is running, starting it if needed. */
export function ensureContainerRuntimeRunning(): void {
  try {
    execSync(`${CONTAINER_RUNTIME_BIN} info`, {
      stdio: 'pipe',
      timeout: 10000,
    });
    log.debug('Container runtime already running');
  } catch (err) {
    log.error('Failed to reach container runtime', { err });
    console.error('\n╔════════════════════════════════════════════════════════════════╗');
    console.error('║  FATAL: Container runtime failed to start                      ║');
    console.error('║                                                                ║');
    console.error('║  Agents cannot run without a container runtime. To fix:        ║');
    console.error('║  1. Ensure Docker is installed and running                     ║');
    console.error('║  2. Run: docker info                                           ║');
    console.error('║  3. Restart NanoClaw                                           ║');
    console.error('╚════════════════════════════════════════════════════════════════╝\n');
    throw new Error('Container runtime is required but failed to start', {
      cause: err,
    });
  }
}

/**
 * Kill orphaned NanoClaw containers from THIS install's previous runs.
 *
 * Scoped by label `nanoclaw-install=<slug>` so a crash-looping peer install
 * cannot reap our containers, and we cannot reap theirs. The label is
 * stamped onto every container at spawn time — see container-runner.ts.
 */
export function cleanupOrphans(): void {
  try {
    const output = execSync(
      `${CONTAINER_RUNTIME_BIN} ps --filter label=${CONTAINER_INSTALL_LABEL} --format '{{.Names}}'`,
      {
        stdio: ['pipe', 'pipe', 'pipe'],
        encoding: 'utf-8',
      },
    );
    const orphans = output.trim().split('\n').filter(Boolean);
    for (const name of orphans) {
      try {
        stopContainer(name);
      } catch {
        /* already stopped */
      }
    }
    if (orphans.length > 0) {
      log.info('Stopped orphaned containers', { count: orphans.length, names: orphans });
    }
  } catch (err) {
    log.warn('Failed to clean up orphaned containers', { err });
  }
}
