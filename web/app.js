// Estado e roteamento. Nenhuma fórmula aqui: o JSON já traz tudo calculado.
import * as api from "./api.js";
import { h, section, fixtureBanner, renderDiagnostic, renderRecommendationBody, renderRejected, renderNext, taskLabel } from "./render.js";

// O fragment com o token precisa sumir da URL antes de qualquer outra coisa.
const pairedByLink = api.consumeFragment();

const $ = (id) => document.getElementById(id);
const MAX_BYTES = 8 * 1024 * 1024;
const state = { doc: null, kind: null, mode: null, filters: { task: "all", commercial: false } };

// ---------- status / erros ----------
function status(msg, level = "info") {
  const box = $("status");
  box.replaceChildren();
  if (!msg) return;
  box.append(h("div", { class: `banner banner-${level === "error" ? "error" : level === "warn" ? "warn" : "info"}`, role: level === "error" ? "alert" : null }, msg));
}

// ---------- validação defensiva ----------
const isObj = (x) => x !== null && typeof x === "object" && !Array.isArray(x);
const isInt = (x) => Number.isInteger(x);

export function validate(doc) {
  if (!isObj(doc)) return { error: "O conteúdo não é um objeto JSON (esperado um system.json ou feasibility.json)." };
  if (doc.schema_version !== 1) return { error: `schema_version ${JSON.stringify(doc.schema_version)} não suportado: esta página entende a versão 1. Gere o arquivo de novo com a CLI atual.` };
  if (doc.kind === "system") {
    for (const k of ["cpu", "memory", "os"]) if (!isObj(doc[k])) return { error: `system.json inválido: falta o bloco "${k}".` };
    return { kind: "system" };
  }
  if (doc.kind !== "feasibility") return { error: `kind ${JSON.stringify(doc.kind)} não reconhecido: esperado "feasibility" (ou "system").` };
  for (const k of ["candidates", "rejected", "pools", "warnings"]) if (!Array.isArray(doc[k])) return { error: `feasibility.json inválido: "${k}" deveria ser uma lista.` };
  if (!isObj(doc.inputs) || !isObj(doc.inputs.intent) || !Array.isArray(doc.inputs.intent.tasks)) return { error: 'feasibility.json inválido: falta "inputs.intent.tasks".' };
  if (!isObj(doc.inputs.system) || doc.inputs.system.kind !== "system") return { error: 'feasibility.json inválido: falta o system embutido em "inputs.system".' };
  for (const [i, c] of doc.candidates.entries()) {
    if (!isObj(c) || !isObj(c.model) || typeof c.model.id !== "string" || !isObj(c.quant) || !isInt(c.total_bytes) || !Array.isArray(c.by_pool)) return { error: `Candidato #${i} malformado (model, quant, total_bytes ou by_pool).` };
    if (c.tokens_per_s_est !== null && typeof c.tokens_per_s_est !== "number") return { error: `Candidato #${i}: tokens_per_s_est deveria ser número ou null.` };
    if (c.tokens_per_s_est_start !== undefined && c.tokens_per_s_est_start !== null && typeof c.tokens_per_s_est_start !== "number") return { error: `Candidato #${i}: tokens_per_s_est_start deveria ser número ou null.` };
  }
  for (const [i, r] of doc.rejected.entries()) if (!isObj(r) || !isObj(r.model) || typeof r.model.id !== "string" || typeof r.message !== "string") return { error: `Rejeitado #${i} malformado.` };
  for (const [i, p] of doc.pools.entries()) if (!isObj(p) || typeof p.id !== "string" || !isInt(p.total_bytes)) return { error: `Pool #${i} malformado.` };
  return { kind: "feasibility" };
}

// ---------- render ----------
function recommendationSection(doc) {
  const body = h("div", { id: "rec-body" });
  const paint = () => body.replaceChildren(...renderRecommendationBody(doc, state.filters));
  const sel = h("select", { id: "f-task" }, h("option", { value: "all" }, "Todas as tarefas"), doc.inputs.intent.tasks.map((t) => h("option", { value: t }, taskLabel(t))));
  sel.value = state.filters.task;
  sel.addEventListener("change", () => { state.filters.task = sel.value; paint(); });
  const chk = h("input", { type: "checkbox", id: "f-comm" });
  chk.checked = state.filters.commercial;
  chk.addEventListener("change", () => { state.filters.commercial = chk.checked; paint(); });
  const info = doc.inputs.intent;
  paint();
  return section("h-rec", "3. Recomendações",
    h("p", { class: "muted" }, `Contexto ${info.ctx} tokens, KV ${info.kv_quant}, batch ${info.batch}. Para outro contexto, rode a CLI de novo ou use o modo LIVE.`),
    h("div", { class: "filters" }, h("span", {}, h("label", { for: "f-task" }, "Tarefa: "), sel), h("span", {}, chk, " ", h("label", { for: "f-comm" }, "Uso comercial (esconde modelos sem licença comercial confirmada)"))),
    h("p", { class: "muted small" }, "Legenda: ", h("span", { class: "prov prov-catalog" }, "catálogo"), " ", h("span", { class: "prov prov-measured" }, "medido"), " ", h("span", { class: "prov prov-calculated" }, "calculado"), " ", h("span", { class: "prov prov-estimated" }, "estimado"), " (valores com ~ são estimativas, não garantias)."),
    body);
}

function render() {
  const out = $("resultados");
  out.replaceChildren();
  const { doc, kind } = state;
  if (!doc) return;
  try {
    out.append(h("div", { class: "banner banner-info", role: "note" }, `Fonte: ${state.mode}. Gerado pela CLI scout ${doc.scout_version ?? "?"}.`));
    const fb = fixtureBanner(doc);
    if (fb) out.append(fb);
    out.append(renderDiagnostic(doc, kind));
    if (kind === "feasibility") out.append(recommendationSection(doc), renderRejected(doc), renderNext(doc, (m) => status(m)));
  } catch (e) {
    out.replaceChildren(h("div", { class: "banner banner-error", role: "alert" }, "Não consegui exibir este arquivo: ", String(e && e.message ? e.message : e), ". Confira se ele foi gerado pela CLI atual."));
  }
}

function setModeLabel(text) { $("modo").textContent = text; }

function load(obj, mode) {
  const v = validate(obj);
  if (v.error) {
    state.doc = null;
    $("resultados").replaceChildren();
    status(v.error, "error");
    return false;
  }
  state.doc = obj; state.kind = v.kind; state.mode = mode;
  setModeLabel(mode);
  status(v.kind === "system" ? "system.json lido: só o Diagnóstico está disponível." : "feasibility.json lido.");
  render();
  $("resultados").focus({ preventScroll: false });
  return true;
}

// ---------- modo ARQUIVO ----------
function parseText(text, origin) {
  if (text.length > MAX_BYTES) return status(`${origin}: arquivo grande demais (limite de 8 MB).`, "error");
  let obj;
  try { obj = JSON.parse(text); } catch (e) { return status(`${origin}: não é JSON válido (${e.message}).`, "error"); }
  load(obj, "ARQUIVO");
}

async function readFile(file) {
  if (!file) return;
  if (file.size > MAX_BYTES) return status("Arquivo grande demais (limite de 8 MB).", "error");
  try { parseText(await file.text(), file.name); } catch { status("Não consegui ler o arquivo.", "error"); }
}

async function loadFixture(name) {
  try {
    const res = await fetch(`fixtures/${name}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    parseText(await res.text(), name);
  } catch (e) {
    status(`Não consegui carregar o exemplo (${e.message}). Abra a página por um servidor HTTP (python3 -m http.server -d web 5173), não por file://.`, "error");
  }
}

// ---------- modo LIVE ----------
let busy = false;
async function live(refresh) {
  if (busy) return;
  const typed = $("token").value;
  if (typed) { api.setToken(typed); $("token").value = ""; }
  api.setPort($("port").value);
  if (!api.hasToken()) return status("Falta o token. Rode `scout serve` e abra o link impresso no terminal, ou cole o token no campo.", "error");
  const ctx = Number($("ctx").value);
  if (!Number.isInteger(ctx) || ctx < 1) return status("Contexto inválido: use um inteiro positivo de tokens.", "error");
  busy = true;
  $("btn-live").disabled = $("btn-refresh").disabled = true;
  try {
    status("Conectando ao backend local...");
    await api.health();
    status(refresh ? "Recoletando o hardware..." : "Lendo o hardware (a primeira coleta pode demorar)...");
    await api.getSystem(refresh);
    status("Calculando recomendações...");
    const plan = await api.postPlan({ ctx, commercial_use: false, batch: 1, kv_quant: "f16" });
    if (load(plan, "LIVE")) status("Conectado ao backend local (127.0.0.1). Resultado atualizado.");
  } catch (e) {
    const msg = e instanceof api.ApiError ? e.message : `Erro inesperado: ${e && e.message ? e.message : e}`;
    status(msg, "error");
    if (e instanceof api.ApiError && e.kind === "unauthorized") api.setToken(null);
  } finally {
    busy = false;
    $("btn-live").disabled = $("btn-refresh").disabled = false;
  }
}

// ---------- ligações ----------
function init() {
  const drop = $("drop"), file = $("file");
  drop.addEventListener("click", () => file.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => { readFile(file.files[0]); file.value = ""; });
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (e) => readFile(e.dataTransfer && e.dataTransfer.files[0]));
  $("btn-paste").addEventListener("click", () => { const t = $("paste").value.trim(); t ? parseText(t, "Texto colado") : status("Cole o conteúdo do JSON primeiro.", "error"); });
  $("btn-demo").addEventListener("click", () => loadFixture("demo-feasibility.json"));
  $("btn-demo2").addEventListener("click", () => loadFixture("demo-feasibility-rejeitados.json"));
  $("btn-demo4").addEventListener("click", () => loadFixture("demo-feasibility-falta.json"));
  $("btn-demo5").addEventListener("click", () => loadFixture("demo-feasibility-nota.json"));
  $("btn-demo8").addEventListener("click", () => loadFixture("demo-feasibility-calibracao-descartada.json"));
  $("btn-demo3").addEventListener("click", () => loadFixture("demo-system.json"));
  $("btn-demo6").addEventListener("click", () => loadFixture("demo-feasibility-calibracao.json"));
  $("btn-demo7").addEventListener("click", () => loadFixture("demo-feasibility-calibracao-na.json"));
  $("btn-live").addEventListener("click", () => live(false));
  $("btn-refresh").addEventListener("click", () => live(true));
  $("port").value = String(api.getPort()); // porta do fragment tem precedência sobre o padrão do campo
  if (pairedByLink) { status("Token recebido pelo link e removido da barra de endereço."); live(false); }
}

init();
