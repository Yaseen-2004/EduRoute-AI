"""
"Knowledge galaxy": the animated 3D hero at the top of the app.

The 26 engineering branch groups float as glowing nodes on a slowly turning 3D
sphere around a pulsing core (the router). Pulses of light travel from the core
to the branches, a picture of questions being routed to knowledge. Students can
drag to spin it and click a branch to focus the assistant on it (the choice
syncs with the sidebar). Clicking the core clears the branch.

Pure canvas 2D with a hand-written perspective projection: no external
libraries, so it works offline and on any host. It pauses when off-screen or
in a background tab, and stays still for users who prefer reduced motion.
"""

from collections.abc import Callable

import streamlit as st

HTML = """
<div class="galaxy">
  <canvas aria-hidden="true"></canvas>
  <div class="copy">
    <div class="eyebrow">Adaptive AI study assistant</div>
    <h1>EduRoute <span>AI</span></h1>
    <p class="tagline"></p>
    <div class="chips"></div>
    <div class="hint">Drag to spin · click a branch to focus</div>
  </div>
  <div class="tip" role="status"></div>
</div>
"""

CSS = """
.galaxy { position: relative; width: 100%; height: 300px; overflow: hidden; cursor: grab;
  border-radius: 1rem; isolation: isolate;
  background: radial-gradient(120% 140% at 75% 40%, #1d1650 0%, #0d1233 45%, #070a1c 100%);
  box-shadow: 0 10px 40px -12px rgba(76, 29, 149, .55), inset 0 0 0 1px rgba(148, 163, 255, .12); }
.galaxy.dragging { cursor: grabbing; }
canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.copy { position: absolute; left: clamp(16px, 4%, 40px); top: 50%; transform: translateY(-50%);
  max-width: min(46%, 440px); color: #eef0ff; pointer-events: none; z-index: 2;
  text-shadow: 0 2px 18px rgba(7, 10, 28, .9); }
.eyebrow { font: 600 .72rem/1 var(--st-font, Inter, sans-serif); letter-spacing: .18em;
  text-transform: uppercase; color: #a5b4fc; margin-bottom: .55rem; }
h1 { margin: 0; font: 700 clamp(1.9rem, 4.2vw, 3rem)/1.05 var(--st-heading-font, 'Space Grotesk', sans-serif);
  letter-spacing: -.02em; color: #fff; }
h1 span { background: linear-gradient(90deg, #c4b5fd, #67e8f9 55%, #f5d0fe);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  text-shadow: none; filter: drop-shadow(0 0 14px rgba(103, 232, 249, .35)); }
.tagline { margin: .7rem 0 .9rem; font: 400 .95rem/1.45 var(--st-font, Inter, sans-serif); color: #cfd5f6; }
.chips { display: flex; flex-wrap: wrap; gap: .4rem; }
.chips span { font: 600 .74rem/1 var(--st-font, Inter, sans-serif); color: #e0e7ff;
  padding: .38rem .62rem; border-radius: 999px; background: rgba(139, 92, 246, .18);
  border: 1px solid rgba(167, 139, 250, .35); backdrop-filter: blur(4px); }
.hint { margin-top: .8rem; font: 400 .72rem/1.3 var(--st-font, Inter, sans-serif); color: #8f9bd1; }
.compact .tagline, .compact .chips, .compact .hint { display: none; }
.compact h1 { font-size: clamp(1.4rem, 3vw, 1.9rem); }
.compact .eyebrow { margin-bottom: .35rem; }
.tip { position: absolute; z-index: 3; pointer-events: none; opacity: 0; transition: opacity .15s;
  font: 500 .78rem/1.3 var(--st-font, Inter, sans-serif); color: #fff; padding: .4rem .6rem;
  border-radius: .5rem; background: rgba(15, 18, 48, .92); border: 1px solid rgba(167, 139, 250, .5);
  white-space: nowrap; }
.tip.on { opacity: 1; }
@media (max-width: 640px) {
  .copy { top: 16px; transform: none; max-width: 92%; }
  .tagline, .hint { display: none; }
}
"""

JS = r"""
const INSTANCES = new WeakMap();

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let g = INSTANCES.get(parentElement);
  if (!g) { g = createGalaxy(parentElement); INSTANCES.set(parentElement, g); }
  g.trigger = setTriggerValue;
  g.update(data || {});
  return () => { g.destroy(); INSTANCES.delete(parentElement); };
}

function createGalaxy(root) {
  const wrap = root.querySelector(".galaxy");
  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const tip = root.querySelector(".tip");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const g = { nodes: [], selected: null, yaw: 0.6, pitch: 0.38, vyaw: 0, vpitch: 0,
              pulses: [], stars: [], w: 0, h: 0, dpr: 1, running: true, visible: true,
              hover: -1, last: performance.now(), spawn: 0, raf: 0, trigger: null };

  // ------------------------------------------------------------- layout
  function resize() {
    const r = wrap.getBoundingClientRect();
    g.dpr = Math.min(window.devicePixelRatio || 1, 2);
    g.w = Math.max(1, r.width); g.h = Math.max(1, r.height);
    canvas.width = Math.round(g.w * g.dpr); canvas.height = Math.round(g.h * g.dpr);
    ctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0);
    g.stars = Array.from({ length: Math.round(g.w * g.h / 2600) }, () => ({
      x: Math.random() * g.w, y: Math.random() * g.h, r: Math.random() * 1.2 + 0.2,
      p: Math.random() * Math.PI * 2, s: 0.4 + Math.random() * 1.4 }));
    if (reduced) draw(0);
  }
  const ro = new ResizeObserver(resize); ro.observe(wrap);

  function centre() {
    const narrow = g.w < 640;
    return { cx: narrow ? g.w * 0.5 : g.w * 0.70, cy: narrow ? g.h * 0.62 : g.h * 0.52,
             R: Math.min(narrow ? g.w * 0.30 : g.w * 0.23, g.h * 0.40) };
  }

  // Fibonacci sphere: evenly spread branch nodes.
  function placeNodes(branches) {
    const n = branches.length, golden = Math.PI * (3 - Math.sqrt(5));
    g.nodes = branches.map((b, i) => {
      const y = 1 - (i / Math.max(1, n - 1)) * 2, rad = Math.sqrt(1 - y * y), th = golden * i;
      const hue = (i * 137.508 + 250) % 360;
      return { ...b, x: Math.cos(th) * rad, y: y * 0.92, z: Math.sin(th) * rad, hue, sx: 0, sy: 0, sz: 0, size: 0 };
    });
  }

  // ------------------------------------------------------------ 3D maths
  function project(x, y, z, c) {
    const cy = Math.cos(g.yaw), sy = Math.sin(g.yaw), cp = Math.cos(g.pitch), sp = Math.sin(g.pitch);
    let x1 = x * cy - z * sy, z1 = x * sy + z * cy;
    let y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
    const f = 2.6 / (2.6 + z2);                 // perspective
    return { x: c.cx + x1 * c.R * f, y: c.cy + y1 * c.R * f, z: z2, f };
  }

  // ------------------------------------------------------------- drawing
  function draw(t) {
    const c = centre();
    ctx.clearRect(0, 0, g.w, g.h);

    for (const s of g.stars) {                       // twinkling starfield
      const a = 0.25 + 0.35 * (1 + Math.sin(t * 0.001 * s.s + s.p)) / 2;
      ctx.fillStyle = `rgba(210,220,255,${a})`;
      ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2); ctx.fill();
    }

    // orbit rings
    ctx.save(); ctx.translate(c.cx, c.cy);
    for (const [k, a] of [[1.25, 0.10], [1.55, 0.06]]) {
      ctx.strokeStyle = `rgba(167,139,250,${a})`; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.ellipse(0, 0, c.R * k, c.R * k * 0.32, -0.25, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.restore();

    for (const n of g.nodes) {
      const p = project(n.x, n.y, n.z, c);
      n.sx = p.x; n.sy = p.y; n.sz = p.z; n.size = (n.code === g.selected ? 7.5 : 4.2) * p.f;
    }
    const order = [...g.nodes.keys()].sort((a, b) => g.nodes[b].sz - g.nodes[a].sz);

    // spokes from the core
    for (const n of g.nodes) {
      const depth = (1 - n.sz) / 2, sel = n.code === g.selected;
      ctx.strokeStyle = `hsla(${n.hue},90%,70%,${sel ? 0.55 : 0.06 + 0.12 * depth})`;
      ctx.lineWidth = sel ? 1.6 : 0.8;
      ctx.beginPath(); ctx.moveTo(c.cx, c.cy); ctx.lineTo(n.sx, n.sy); ctx.stroke();
    }

    // core: the router
    const pulse = reduced ? 1 : 1 + 0.08 * Math.sin(t * 0.003);
    const coreR = c.R * 0.16 * pulse;
    let grad = ctx.createRadialGradient(c.cx, c.cy, 0, c.cx, c.cy, coreR * 3.2);
    grad.addColorStop(0, "rgba(255,255,255,0.95)"); grad.addColorStop(0.18, "rgba(196,181,253,0.9)");
    grad.addColorStop(0.45, "rgba(124,58,237,0.35)"); grad.addColorStop(1, "rgba(124,58,237,0)");
    ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(c.cx, c.cy, coreR * 3.2, 0, Math.PI * 2); ctx.fill();

    // routing pulses travelling out along curved paths
    for (const q of g.pulses) {
      const n = g.nodes[q.i]; if (!n) continue;
      const e = q.t < 1 ? q.t : 1, lift = Math.sin(e * Math.PI) * 0.25;
      const p = project(n.x * e, n.y * e - lift, n.z * e, c);
      const a = q.t < 1 ? 0.95 : Math.max(0, 1 - (q.t - 1) * 3);
      grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 6 * p.f);
      grad.addColorStop(0, `hsla(${n.hue},100%,85%,${a})`); grad.addColorStop(1, `hsla(${n.hue},100%,70%,0)`);
      ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(p.x, p.y, 6 * p.f, 0, Math.PI * 2); ctx.fill();
    }

    // nodes, back to front, with glow and labels
    for (const idx of order) {
      const n = g.nodes[idx], front = n.sz < 0, sel = n.code === g.selected, hov = idx === g.hover;
      const depth = (1 - n.sz) / 2, r = n.size * (hov ? 1.35 : 1);
      grad = ctx.createRadialGradient(n.sx, n.sy, 0, n.sx, n.sy, r * 4);
      grad.addColorStop(0, `hsla(${n.hue},95%,${sel ? 80 : 72}%,${0.35 + 0.55 * depth})`);
      grad.addColorStop(1, `hsla(${n.hue},95%,60%,0)`);
      ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(n.sx, n.sy, r * 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = `hsla(${n.hue},100%,${88 - 10 * (1 - depth)}%,${0.55 + 0.45 * depth})`;
      ctx.beginPath(); ctx.arc(n.sx, n.sy, r, 0, Math.PI * 2); ctx.fill();
      if (sel) {
        ctx.strokeStyle = `hsla(${n.hue},100%,85%,0.9)`; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(n.sx, n.sy, r + 5 + (reduced ? 0 : 2 * Math.sin(t * 0.006)), 0, Math.PI * 2); ctx.stroke();
      }
      if ((front && !g.compact) || sel || hov) {        // compact banner: labels on hover/selection only
        ctx.font = `${sel || hov ? 600 : 500} ${sel ? 12 : 10.5}px Inter, sans-serif`;
        ctx.fillStyle = `rgba(230,234,255,${sel || hov ? 1 : 0.35 + 0.6 * depth})`;
        ctx.fillText(n.code, n.sx + r + 4, n.sy + 3.5);
      }
    }
  }

  // ------------------------------------------------------------ animation
  function frame(now) {
    g.raf = 0;
    if (!g.running || !g.visible) return;
    const dt = Math.min(50, now - g.last); g.last = now;
    if (!g.dragging) {
      g.yaw += dt * 0.00012 + g.vyaw; g.pitch += g.vpitch;
      g.vyaw *= 0.94; g.vpitch *= 0.94;
      g.pitch = Math.max(-0.9, Math.min(0.9, g.pitch * 0.995 + 0.38 * 0.005));
    }
    g.spawn -= dt;
    if (g.spawn <= 0 && g.nodes.length) {
      const selIdx = g.nodes.findIndex(n => n.code === g.selected);
      const i = selIdx >= 0 && Math.random() < 0.55 ? selIdx : Math.floor(Math.random() * g.nodes.length);
      g.pulses.push({ i, t: 0, v: 0.0007 + Math.random() * 0.0005 });
      g.spawn = 260 + Math.random() * 260;
    }
    for (const q of g.pulses) q.t += q.v * dt;
    g.pulses = g.pulses.filter(q => q.t < 1.35);
    draw(now);
    g.raf = requestAnimationFrame(frame);
  }
  function start() { if (!reduced && !g.raf && g.running && g.visible) { g.last = performance.now(); g.raf = requestAnimationFrame(frame); } }

  const io = new IntersectionObserver(([e]) => { g.visible = e.isIntersecting; start(); });
  io.observe(wrap);
  const onVis = () => { g.visible = !document.hidden; start(); };
  document.addEventListener("visibilitychange", onVis);

  // ---------------------------------------------------------- interaction
  function nodeAt(x, y) {
    let best = -1, bestD = 16;
    g.nodes.forEach((n, i) => { const d = Math.hypot(n.sx - x, n.sy - y); if (d < Math.max(bestD, n.size * 2) && d < bestD + n.size) { best = i; bestD = d; } });
    return best;
  }
  function local(e) { const r = wrap.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  let down = null, moved = false;
  wrap.onpointerdown = (e) => { down = local(e); moved = false; g.dragging = true; wrap.classList.add("dragging");
    wrap.setPointerCapture(e.pointerId); };
  wrap.onpointermove = (e) => {
    const p = local(e);
    if (down) {
      const dx = p.x - down.x, dy = p.y - down.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      g.yaw += dx * 0.008; g.pitch = Math.max(-0.9, Math.min(0.9, g.pitch + dy * 0.006));
      g.vyaw = dx * 0.0016; g.vpitch = dy * 0.0008; down = p;
      if (reduced) draw(0);
      return;
    }
    const i = nodeAt(p.x, p.y); g.hover = i;
    const c = centre(), onCore = Math.hypot(p.x - c.cx, p.y - c.cy) < c.R * 0.3;
    wrap.style.cursor = i >= 0 || onCore ? "pointer" : "grab";
    if (i >= 0) {
      const n = g.nodes[i];
      tip.textContent = `${n.name} · ${n.subjects} subjects` + (n.code === g.selected ? " (focused)" : "");
      tip.style.left = Math.min(g.w - 220, p.x + 14) + "px"; tip.style.top = Math.max(6, p.y - 34) + "px";
      tip.classList.add("on");
    } else if (onCore) {
      tip.textContent = "Router core · click for all branches";
      tip.style.left = Math.min(g.w - 220, p.x + 14) + "px"; tip.style.top = Math.max(6, p.y - 34) + "px";
      tip.classList.add("on");
    } else tip.classList.remove("on");
    if (reduced) draw(0);
  };
  wrap.onpointerleave = () => { g.hover = -1; tip.classList.remove("on"); };
  wrap.onpointerup = (e) => {
    const p = local(e);
    g.dragging = false; wrap.classList.remove("dragging"); down = null;
    if (moved || !g.trigger) return;
    const i = nodeAt(p.x, p.y), c = centre();
    if (i >= 0) g.trigger("branch", g.nodes[i].code === g.selected ? "" : g.nodes[i].code);
    else if (Math.hypot(p.x - c.cx, p.y - c.cy) < c.R * 0.3) g.trigger("branch", "");
  };

  // --------------------------------------------------------------- data
  g.update = (d) => {
    g.compact = !!d.compact;
    // Streamlit wraps the component in containers without a height, so the
    // galaxy gets its pixel height explicitly (a percentage would resolve to 0).
    wrap.style.height = `${d.height || (g.compact ? 150 : 300)}px`;
    wrap.classList.toggle("compact", g.compact);
    root.querySelector(".tagline").textContent = d.tagline || "";
    root.querySelector(".chips").replaceChildren(...(d.chips || []).map(t => {
      const s = document.createElement("span"); s.textContent = t; return s; }));
    const codes = (d.branches || []).map(b => b.code).join(",");
    if (codes !== g.codes) { placeNodes(d.branches || []); g.codes = codes; }
    g.selected = d.selected || null;
    resize();
    start();
    if (reduced) draw(0);
  };
  g.destroy = () => { g.running = false; if (g.raf) cancelAnimationFrame(g.raf);
    ro.disconnect(); io.disconnect(); document.removeEventListener("visibilitychange", onVis); };
  return g;
}
"""

_GALAXY = st.components.v2.component("eduroute_knowledge_galaxy", html=HTML, css=CSS, js=JS)


def knowledge_galaxy(branches, selected=None, *, compact=False, tagline="", chips=(),
                     key="galaxy", on_branch_change: Callable[[], None] | None = None):
    """Render the hero. `branches`: [{"code", "name", "subjects"}]. Returns the
    component result; `result.branch` is the clicked branch code ("" = all)."""
    height = 150 if compact else 300
    return _GALAXY(
        data={"branches": list(branches), "selected": selected, "compact": compact,
              "tagline": tagline, "chips": list(chips), "height": height},
        key=key, height=height,
        on_branch_change=on_branch_change or (lambda: None),
    )
