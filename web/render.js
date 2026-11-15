// Renderização. Só formatação: nenhum cálculo de memória, tokens/s ou tier.
// Todo dado do JSON entra por textContent / createElement, nunca innerHTML.

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// ---- formatação (igual a fmt_gb / fmt_gbps da CLI) ----
export const fmtGiB = (n) => (Number.isFinite(n) ? `${(n / 2 ** 30).toFixed(2)} GiB` : "—");
export const fmtGBps = (n) => (Number.isFinite(n) ? `${(n / 1e9).toFixed(1)} GB/s` : "—");
const fmtTps = (n) => `${Number(n).toFixed(1)} tok/s`;

const PROV = {
  catalog: ["catálogo", "dado do catálogo, com fonte"],
  measured: ["medido", "lido da máquina"],
  calculated: ["calculado", "fórmula determinística sobre dados conhecidos"],
  estimated: ["estimado", "depende de constante empírica ou de entrada de baixa confiança"],
};
const PLACEMENT = { gpu_full: "GPU inteira", hybrid: "híbrido GPU+CPU", cpu_only: "só CPU" };
const TIER = { confortavel: "confortável", limite: "no limite", inviavel: "inviável" };
const CONF = { high: "alta", medium: "média", low: "baixa" };
const CONF_REASON = {
  weights_fallback: "tamanho dos pesos aproximado",
  bpw_from_table: "bits por peso vindos de tabela",
  bandwidth_estimated: "banda de memória estimada",
  bandwidth_unknown: "banda de memória desconhecida",
  hardware_low_confidence: "hardware lido com baixa confiança",
  eta_default: "eficiência padrão, não calibrada nesta máquina",
  ttft_unspecified: "tempo até o primeiro token não especificado",
};
const REJECT = {
  ctx_exceeds_max: "contexto acima do máximo do modelo",
  kv_model_unsupported: "cache KV não suportado para este modelo",
  below_quality_floor: "abaixo do piso de qualidade",
  no_usable_quant: "nenhuma quantização utilizável",
  does_not_fit: "não cabe na memória",
  format_requires_full_gpu: "formato exige o modelo inteiro na GPU",
  no_gpu_for_format: "formato exige GPU e não há uma",
};
const BLOCK = { non_commercial: "licença não comercial", license_unknown: "licença desconhecida" };
const TASK = { chat: "Chat", code: "Código", embeddings: "Embeddings", vlm: "Visão", stt: "Voz para texto", diffusion: "Imagem" };
export const taskLabel = (t) => TASK[t] || String(t);

export function provBadge(kind) {
  const [label, tip] = PROV[kind] || ["desconhecida", "proveniência não informada"];
  const k = PROV[kind] ? kind : "unknown";
  return h("span", { class: `prov prov-${k}`, title: tip }, label);
}

// Número + selo de proveniência. Estimado leva "~" e borda tracejada.
function num(text, kind) {
  return h("span", { class: "num" }, h("span", { class: kind === "estimated" ? "est-val" : "val" }, kind === "estimated" ? `~${text}` : text), " ", provBadge(kind));
}

const safeUrl = (u) => (typeof u === "string" && /^https?:\/\//i.test(u) ? u : null);

function srcLink(url, text) {
  const u = safeUrl(url);
  return u ? h("a", { href: u, rel: "noopener noreferrer", target: "_blank" }, text) : h("span", {}, text);
}

const fieldVal = (f) => (f && typeof f === "object" && "value" in f ? f.value : undefined);

function fieldText(f, fmt = (x) => String(x)) {
  const v = fieldVal(f);
  if (v === null || v === undefined) return h("em", { class: "unknown" }, "não obtido");
  const out = [fmt(v)];
  if (f.confidence && f.confidence !== "high") out.push(" ", h("span", { class: "conf conf-" + f.confidence, title: "confiança da leitura" }, `confiança ${CONF[f.confidence] || f.confidence}`));
  return h("span", {}, out);
}

export function table(caption, headers, rows) {
  return h("table", {},
    h("caption", {}, caption),
    h("thead", {}, h("tr", {}, headers.map((t) => h("th", { scope: "col" }, t)))),
    h("tbody", {}, rows));
}

export function section(id, title, ...kids) {
  return h("section", { "aria-labelledby": id }, h("h2", { id }, title), ...kids);
}

export function fixtureBanner(doc) {
  const fx = doc.fixture || (doc.inputs && doc.inputs.fixture) || (doc.inputs && doc.inputs.system && doc.inputs.system.fixture);
  if (!fx || !fx.synthetic) return null;
  return h("div", { class: "banner banner-warn", role: "note" }, h("strong", {}, "DADOS SINTÉTICOS. "), String(fx.description || "Nenhum valor é medição real."));
}

// ---------- Diagnóstico ----------
function hardwareList(sys) {
  const cpu = sys.cpu || {}, mem = sys.memory || {}, os = sys.os || {}, pw = sys.power || {};
  const gpus = fieldVal(sys.gpus);
  const disks = fieldVal(sys.disks);
  const rt = sys.runtimes || {};
  const items = [
    ["Sistema", h("span", {}, [fieldText(os.name), " ", fieldText(os.version)])],
    ["CPU", h("span", {}, [fieldText(cpu.model), " · ", fieldText(cpu.cores_physical), " núcleos físicos, ", fieldText(cpu.cores_logical), " lógicos"])],
    ["RAM", h("span", {}, [fieldText(mem.ram_total_bytes, fmtGiB), " ", fieldText(mem.type), " ", fieldText(mem.speed_mts, (x) => `${x} MT/s`)])],
    ["Banda de RAM", fieldText(mem.bandwidth_bytes_per_s, fmtGBps)],
    ["Memória unificada", fieldText(mem.unified, (x) => (x ? "sim" : "não"))],
    ["Swap", fieldText(mem.swap_total_bytes, fmtGiB)],
  ];
  if (Array.isArray(gpus) && gpus.length) {
    gpus.forEach((g, i) => items.push([`GPU ${i}`, h("span", {}, [fieldText(g.name), " · VRAM ", fieldText(g.vram_total_bytes, fmtGiB), " · banda ", fieldText(g.bandwidth_bytes_per_s, fmtGBps), " · ", fieldText(g.backend)])]));
  } else {
    items.push(["GPU", h("em", { class: "unknown" }, gpus === null || gpus === undefined ? "não obtido" : "nenhuma detectada")]);
  }
  if (Array.isArray(disks)) disks.forEach((d) => items.push([`Disco ${d.mount ?? ""}`, h("span", {}, [fieldText(d.kind), " · livre ", fieldText(d.free_bytes, fmtGiB), " de ", fieldText(d.total_bytes, fmtGiB)])]));
  items.push(["Bateria", fieldText(pw.on_battery, (x) => (x ? "na bateria" : "na tomada"))]);
  const present = Object.entries(rt).filter(([, f]) => fieldVal(f) === true).map(([k]) => k);
  items.push(["Runtimes no PATH", present.length ? present.join(", ") : "nenhum encontrado"]);
  return h("dl", { class: "kv" }, items.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

function poolsTable(pools) {
  const rows = pools.map((p) => {
    const pv = p.provenance || {};
    return h("tr", {},
      h("th", { scope: "row" }, `${p.id}${p.gpu_index !== null && p.gpu_index !== undefined ? ` (GPU ${p.gpu_index})` : ""}`),
      h("td", {}, num(fmtGiB(p.total_bytes), pv.total_bytes || "measured")),
      h("td", {}, num(fmtGiB(p.reserved_bytes), pv.reserved_bytes || "calculated")),
      h("td", {}, num(fmtGiB(p.usable_bytes), pv.usable_bytes || "calculated")),
      h("td", {}, p.bandwidth_bytes_per_s == null ? h("em", { class: "unknown" }, "desconhecida") : num(fmtGBps(p.bandwidth_bytes_per_s), pv.bandwidth_bytes_per_s || "measured")),
      h("td", {}, p.eta == null ? "—" : num(String(p.eta), pv.eta || "estimated")));
  });
  return h("div", { class: "scroll" }, table("Pools de memória", ["Pool", "Total", "Reservado", "Utilizável", "Banda", "Eficiência (eta)"], rows));
}

function warningsList(ws) {
  if (!ws.length) return h("p", {}, "Nenhum aviso.");
  return h("ul", { class: "warnings" }, ws.map((w) => h("li", { class: "warn-" + (w.severity === "warn" ? "warn" : "info") }, h("strong", {}, w.severity === "warn" ? "Atenção: " : "Nota: "), String(w.message))));
}

export function renderDiagnostic(doc, kind) {
  const sys = kind === "system" ? doc : doc.inputs.system;
  const kids = [h("h3", {}, "Hardware"), hardwareList(sys)];
  if (kind === "feasibility") {
    kids.push(poolsTable(doc.pools), h("h3", {}, "Avisos"), warningsList(doc.warnings));
    const t = doc.inputs.tuning;
    if (t) kids.push(h("p", { class: "muted" }, `Eficiência ${t.source === "calibrated" ? "calibrada nesta máquina" : "padrão (não calibrada)"}: GPU ${t.eta_gpu}, CPU ${t.eta_cpu}.`));
  } else {
    kids.push(h("div", { class: "banner banner-info", role: "note" },
      "Este é um system.json: ele só tem o hardware, então só o Diagnóstico aparece. Para ver recomendações, rode ",
      h("code", {}, "scout plan --system system.json --json > feasibility.json"),
      " e envie o feasibility.json (ou use o modo LIVE)."));
  }
  return section("h-diag", "2. Diagnóstico", ...kids);
}

// ---------- Recomendações ----------
function confBadge(c) {
  const low = c.confidence === "low";
  return h("span", { class: "conf conf-" + c.confidence, title: (c.confidence_reasons || []).map((r) => CONF_REASON[r] || r).join("; ") || null }, `${low ? "⚠ " : ""}confiança ${CONF[c.confidence] || c.confidence}`);
}

function tpsCell(c) {
  const t = c.tokens_per_s_est;
  const reasons = (c.confidence_reasons || []).map((r) => CONF_REASON[r] || r);
  if (t === null || t === undefined) {
    return h("td", {}, h("em", { class: "unknown" }, "desconhecido"), h("div", { class: "muted small" }, reasons.length ? `Motivo: ${reasons.join("; ")}.` : "Motivo não informado."));
  }
  const kind = (c.provenance && c.provenance.tokens_per_s_est) || "estimated";
  return h("td", {},
    num(fmtTps(t), kind),
    c.confidence === "low" && h("div", { class: "caveat" }, "⚠ confiança baixa: trate como ordem de grandeza, não como número firme."),
    reasons.length ? h("div", { class: "muted small" }, reasons.join("; ")) : null);
}

function memCell(c) {
  const pv = c.provenance || {};
  const poolsLine = (c.by_pool || []).map((p) => h("li", {}, `${p.pool}: ${fmtGiB(p.total_bytes)} de ${fmtGiB(p.budget_bytes)} (${TIER[p.tier] || p.tier})`));
  return h("td", {},
    num(fmtGiB(c.total_bytes), pv.total_bytes || "calculated"),
    h("details", {}, h("summary", {}, "detalhe"),
      h("ul", { class: "plain" },
        h("li", {}, "pesos ", num(fmtGiB(c.weights_bytes), pv.weights_bytes || "catalog")),
        h("li", {}, "cache KV ", num(fmtGiB(c.kv_bytes), pv.kv_bytes || "calculated")),
        h("li", {}, "buffers ", num(fmtGiB(c.buffer_bytes), pv.buffer_bytes || "calculated")),
        poolsLine)));
}

function licenseCell(m, c) {
  const out = [srcLink(m.license_source, String(m.license ?? "desconhecida")), " ", provBadge("catalog")];
  if (c.blocked) out.push(h("div", { class: "caveat" }, `Bloqueado: ${BLOCK[c.blocked_reason] || "licença"}.`));
  else if (m.commercial_ok === false) out.push(h("div", { class: "caveat" }, "Não permite uso comercial."));
  else if (m.commercial_ok === null || m.commercial_ok === undefined) out.push(h("div", { class: "caveat" }, "Uso comercial: desconhecido."));
  return h("td", {}, out);
}

function candRow(c, task) {
  const m = c.model, q = c.quant || {};
  const primary = (c.primary_for || []).includes(task);
  const pl = [PLACEMENT[c.placement] || c.placement];
  if (c.placement === "hybrid") pl.push(` (${c.n_gpu_layers} de ${m.n_layers ?? "?"} camadas na GPU)`);
  return h("tr", { class: c.blocked ? "row-blocked" : null },
    h("th", { scope: "row" }, srcLink(m.source_url, String(m.id)), primary && h("span", { class: "chip", title: "principal para esta tarefa" }, "principal")),
    h("td", {}, `${q.name} (${q.format}, ${q.bpw} bpw)`),
    h("td", {}, pl, h("div", {}, h("span", { class: "tier tier-" + c.tier }, TIER[c.tier] || c.tier), " ", confBadge(c))),
    memCell(c),
    tpsCell(c),
    h("td", {}, num(fmtGiB(q.file_bytes), "catalog")),
    licenseCell(m, c));
}

// opts: {task: "all"|<task>, commercial: bool}. Só filtra o que o JSON já diz.
export function recommendationGroups(doc, opts) {
  const tasks = (doc.inputs.intent.tasks || []).filter((t) => opts.task === "all" || t === opts.task);
  let hidden = 0;
  const groups = tasks.map((t) => {
    let list = doc.candidates.filter((c) => (c.model.tasks || []).includes(t));
    if (opts.commercial) {
      const keep = list.filter((c) => !c.blocked && c.model.commercial_ok === true);
      hidden += list.length - keep.length;
      list = keep;
    }
    return { task: t, list };
  });
  return { groups, hidden };
}

export function renderRecommendationBody(doc, opts) {
  const { groups, hidden } = recommendationGroups(doc, opts);
  const kids = [];
  if (!doc.candidates.length) kids.push(h("div", { class: "banner banner-warn", role: "note" }, "Nenhum modelo coube nesta máquina com estas configurações. Veja a seção Rejeitados."));
  if (hidden) kids.push(h("p", { class: "muted" }, `${hidden} linha(s) escondida(s) pelo filtro de uso comercial (licença não comercial, bloqueada ou desconhecida).`));
  for (const g of groups) {
    kids.push(h("h3", {}, taskLabel(g.task)));
    if (!g.list.length) { kids.push(h("p", { class: "muted" }, "Nenhuma recomendação para esta tarefa.")); continue; }
    kids.push(h("div", { class: "scroll" }, table(`Recomendações para ${taskLabel(g.task)}`, ["Modelo", "Quantização", "Colocação", "Memória", "Velocidade", "Download", "Licença"], g.list.map((c) => candRow(c, g.task)))));
  }
  return kids;
}

// ---------- Rejeitados ----------
export function renderRejected(doc) {
  const kids = [];
  if (!doc.rejected.length) {
    kids.push(h("p", {}, "Nenhum modelo foi rejeitado: todos os modelos do catálogo para estas tarefas couberam."));
  } else {
    const rows = doc.rejected.map((r) => {
      const cl = r.closest;
      return h("tr", {},
        h("th", { scope: "row" }, srcLink(r.model.source_url, String(r.model.id))),
        h("td", {}, h("strong", {}, REJECT[r.reason_code] || r.reason_code), h("div", {}, String(r.message))),
        h("td", {}, r.missing_bytes == null ? h("em", { class: "unknown" }, "não se aplica") : num(fmtGiB(r.missing_bytes), (r.provenance && r.provenance.missing_bytes) || "calculated")),
        h("td", {}, cl && cl.quant ? `${cl.quant.name} (${PLACEMENT[cl.placement] || cl.placement}), precisaria de ${fmtGiB(cl.total_bytes)}` : "rejeitado antes do cálculo de memória"));
    });
    kids.push(h("div", { class: "scroll" }, table("Modelos que não couberam", ["Modelo", "Motivo", "Quanto faltou", "Tentativa mais próxima"], rows)));
  }
  return section("h-rej", "4. Rejeitados", ...kids);
}

// ---------- Próximo passo ----------
export const benchCommand = (modelId) => `scout bench --run ${modelId}`;

export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); return true; }
  } catch { /* cai no fallback */ }
  const ta = h("textarea", { "aria-hidden": "true", class: "visually-hidden" });
  ta.value = text;
  document.body.append(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch { ok = false; }
  ta.remove();
  return ok;
}

export function renderNext(doc, announce) {
  if (!doc.candidates.length) {
    return section("h-next", "5. Próximo passo", h("p", {}, "Sem candidatos para medir. Reduza o contexto (por exemplo, ", h("code", {}, "scout plan --ctx 4096"), ") ou use outro modelo."));
  }
  const select = h("select", { id: "bench-model" }, doc.candidates.map((c) => h("option", { value: c.model.id }, c.model.id)));
  const first = doc.candidates.find((c) => !c.blocked && (c.primary_for || []).length) || doc.candidates[0];
  select.value = first.model.id;
  const code = h("code", { id: "bench-cmd", class: "cmd" }, benchCommand(select.value));
  select.addEventListener("change", () => { code.textContent = benchCommand(select.value); });
  const btn = h("button", { type: "button" }, "Copiar comando");
  btn.addEventListener("click", async () => announce((await copyText(code.textContent)) ? "Comando copiado." : "Não foi possível copiar; selecione o texto e copie manualmente."));
  return section("h-next", "5. Próximo passo",
    h("p", {}, "Os tokens/s acima são estimativas. Para medir de verdade nesta máquina:"),
    h("div", { class: "row" }, h("label", { for: "bench-model" }, "Modelo"), select),
    h("div", { class: "row" }, code, btn),
    h("p", { class: "muted" }, "Observação: o subcomando ", h("code", {}, "scout bench"), " ainda não executa benchmarks nesta versão da CLI; o comando mostrado é o contrato previsto."));
}
