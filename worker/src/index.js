// Crowd stats for the watermark quiz. KV counters only: no cookies, no
// stored IPs, no user ids. Counter updates are read-modify-write and so
// slightly lossy under concurrency — acceptable here.

import pairs from "../../docs/pairs.json";

const ORIGIN = "https://watermark-quiz.krishmatta.net";
const VALID_QIDS = new Set(pairs.map((p) => p.id));
const DAILY_ANSWER_CAP = 200;

const CORS = {
  "Access-Control-Allow-Origin": ORIGIN,
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

async function bump(kv, key, ttl) {
  const current = parseInt((await kv.get(key)) ?? "0", 10) + 1;
  await kv.put(key, String(current), ttl ? { expirationTtl: ttl } : {});
  return current;
}

async function verifyTurnstile(secret, token, ip) {
  const form = new FormData();
  form.append("secret", secret);
  form.append("response", token);
  if (ip) form.append("remoteip", ip);
  const res = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
    method: "POST",
    body: form,
  });
  if (!res.ok) return false;
  return (await res.json()).success === true;
}

async function ipKey(salt, ip) {
  const day = new Date().toISOString().slice(0, 10);
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${salt}:${ip}:${day}`));
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  return `ip:${hex}`;
}

async function handleAnswer(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "bad json" }, 400);
  }
  const { qid, correct, token } = body ?? {};
  if (!VALID_QIDS.has(qid) || typeof correct !== "boolean" || typeof token !== "string") {
    return json({ error: "bad request" }, 400);
  }

  const ip = request.headers.get("CF-Connecting-IP") ?? "";
  if (!(await verifyTurnstile(env.TURNSTILE_SECRET, token, ip))) {
    return json({ error: "turnstile" }, 403);
  }
  if ((await bump(env.STATS, await ipKey(env.IP_HASH_SALT, ip), 86400)) > DAILY_ANSWER_CAP) {
    return json({ error: "daily cap" }, 429);
  }

  await bump(env.STATS, `q:${qid}:shown`);
  if (correct) await bump(env.STATS, `q:${qid}:correct`);
  return json({ ok: true });
}

async function handleStats(env) {
  const qids = [...VALID_QIDS];
  const counts = await Promise.all(
    qids.flatMap((q) => [env.STATS.get(`q:${q}:shown`), env.STATS.get(`q:${q}:correct`)])
  );
  const questions = {};
  let shown = 0;
  let correct = 0;
  qids.forEach((q, i) => {
    const s = parseInt(counts[2 * i] ?? "0", 10);
    const c = parseInt(counts[2 * i + 1] ?? "0", 10);
    questions[q] = { shown: s, correct: c };
    shown += s;
    correct += c;
  });
  return json({ questions, global: { shown, correct } });
}

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);
    try {
      if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
      if (request.method === "POST" && pathname === "/answer") return await handleAnswer(request, env);
      if (request.method === "GET" && pathname === "/stats") return await handleStats(env);
      return json({ error: "not found" }, 404);
    } catch (err) {
      console.log(JSON.stringify({ level: "error", message: String(err) }));
      return json({ error: "internal" }, 500);
    }
  },
};
