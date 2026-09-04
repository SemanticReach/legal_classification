"use client";

import { useState, useCallback } from "react";

// ── Types ────────────────────────────────────────────────────────────────────

type ClauseType =
  | "Uncapped Liability"
  | "Cap On Liability"
  | "Liquidated Damages"
  | "Non-Compete"
  | "Anti-Assignment"
  | "Change Of Control"
  | "Termination For Convenience"
  | "Ip Ownership Assignment"
  | "Irrevocable Or Perpetual License"
  | "Covenant Not To Sue";

interface SearchHit {
  _score: number;
  data: {
    predicate?: string;
    object?: string;
    contract?: string;
    clause_type?: string;
  };
}

interface ClassifyResult {
  hbPrediction: string;
  hbScore: number;
  vanillaPrediction: string;
  vanillaScore: number;
  allScores: Record<string, number>;
  rescued: boolean;
  peerClauses: SearchHit[];
  riskFlags: string[];
}

// ── Constants ────────────────────────────────────────────────────────────────

const CLAUSE_TYPES: ClauseType[] = [
  "Uncapped Liability",
  "Cap On Liability",
  "Liquidated Damages",
  "Non-Compete",
  "Anti-Assignment",
  "Change Of Control",
  "Termination For Convenience",
  "Ip Ownership Assignment",
  "Irrevocable Or Perpetual License",
  "Covenant Not To Sue",
];

const BENCHMARK_DATA = [
  { model: "HyperBinder (ours)", p1: 94.8, training: "None", compute: "CPU", highlight: true },
  { model: "DeBERTa (fine-tuned)", p1: 87.8, training: "Full fine-tune", compute: "8× A100" },
  { model: "RoBERTa (fine-tuned)", p1: 83.1, training: "Full fine-tune", compute: "8× A100" },
  { model: "BERT (fine-tuned)",    p1: 78.9, training: "Full fine-tune", compute: "8× A100" },
  { model: "GPT-4 (zero-shot)",   p1: 67.2, training: "None",           compute: "API" },
];

const ABLATION_DATA: { type: string; multiSlot: number; vanilla: number }[] = [
  { type: "Irrevocable Or Perpetual License", multiSlot: 98.6, vanilla: 52.9 },
  { type: "Uncapped Liability",               multiSlot: 95.5, vanilla: 67.6 },
  { type: "Liquidated Damages",               multiSlot: 95.1, vanilla: 73.8 },
  { type: "Non-Compete",                      multiSlot: 97.5, vanilla: 82.4 },
  { type: "Covenant Not To Sue",              multiSlot: 98.0, vanilla: 88.0 },
  { type: "Change Of Control",               multiSlot: 90.1, vanilla: 82.6 },
  { type: "Ip Ownership Assignment",          multiSlot: 97.6, vanilla: 91.1 },
  { type: "Anti-Assignment",                  multiSlot: 97.9, vanilla: 97.3 },
];

// Real rescued clauses from CUAD eval — ground truth type + what vanilla got wrong
interface SampleClause {
  text: string;
  trueType: string;
  vanillaWrong: string;
  contract: string;
}

// All samples live-validated against HyperBinder server
// "both-correct" samples first, then rescued samples
const SAMPLE_CLAUSES: Record<string, SampleClause> = {
  "Anti-Assignment ✓": {
    trueType: "Anti-Assignment",
    vanillaWrong: null as unknown as string,
    contract: "Standard clause",
    text: `Neither party may assign this Agreement or any rights or obligations hereunder, by operation of law or otherwise, without the prior written consent of the other party, which consent shall not be unreasonably withheld or delayed.`,
  },
  "Termination ✓": {
    trueType: "Termination For Convenience",
    vanillaWrong: null as unknown as string,
    contract: "Standard clause",
    text: `Either party may terminate this Agreement for any reason or no reason upon thirty (30) days prior written notice to the other party, without liability to the terminating party except for payment of amounts due and owing as of the termination date.`,
  },
  "Irrevocable · AT&T": {
    trueType: "Irrevocable Or Perpetual License",
    vanillaWrong: "License Grant",
    contract: "AT&T / Vendor",
    text: `Vendor hereby grants and promises to grant and have granted to AT&T and its Affiliates a royalty-free, nonexclusive, sublicensable, assignable, transferable, irrevocable, perpetual, world-wide license in and to any applicable Intellectual Property Rights of Vendor to use, copy, modify, distribute, display, perform, import, make, sell, offer to sell, and exploit (and have others do any of the foregoing on or for AT&T's or any of its customers' behalf or benefit) any Intellectual Property Rights of Vendor or any third party that are not included in Material or Paid-For Development but necessary to operate the Cell Sites or receive the full benefit of the Work.`,
  },
  "Irrevocable · Honeywell": {
    trueType: "Irrevocable Or Perpetual License",
    vanillaWrong: "Affiliate License-Licensee",
    contract: "Honeywell / SpinCo",
    text: `Hence, as of the Distribution Date, Honeywell hereby grants, and agrees to cause the members of the Honeywell Group to hereby grant, to SpinCo and the members of the SpinCo Group a non-exclusive, royalty-free, fully-paid, perpetual, sublicenseable, worldwide license to use and exercise rights under the Honeywell Shared IP (excluding Trademarks, the Honeywell Content and the subject matter of any other Ancillary Agreement), said license being limited to use of a similar type, scope and extent as used in the SpinCo Business prior to the Distribution Date and the natural growth and development thereof.`,
  },
};

// ── API Calls ────────────────────────────────────────────────────────────────

async function searchSlots(
  slotQueries: Record<string, { query: string; weight: number; encoding: string }>,
  topK = 10
): Promise<SearchHit[]> {
  const res = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slot_queries: slotQueries, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  const data = await res.json();
  return data.results ?? [];
}

async function runClassify(text: string): Promise<ClassifyResult> {
  // 1. Vanilla search — single global semantic query
  const vanillaHits = await searchSlots({
    object: { query: text.slice(0, 500), weight: 1.0, encoding: "semantic" },
  }, 1);
  const vanillaPrediction = vanillaHits[0]?.data?.predicate ?? "Unknown";
  const vanillaScore = vanillaHits[0]?._score ?? 0;

  // 2. HyperBinder — parallel per-type queries with symbolic filter
  const allScores: Record<string, number> = {};
  await Promise.all(
    CLAUSE_TYPES.map(async (ct) => {
      const hits = await searchSlots({
        clause_type: { query: ct, weight: 0.01, encoding: "exact" },
        object:      { query: text.slice(0, 500), weight: 1.0, encoding: "semantic" },
      }, 3);
      allScores[ct] = hits[0]?._score ?? 0;
    })
  );

  const hbPrediction = Object.entries(allScores).sort((a, b) => b[1] - a[1])[0][0];
  const hbScore = allScores[hbPrediction];

  // 3. Fetch peer clauses for the top prediction
  const peerHits = await searchSlots({
    clause_type: { query: hbPrediction, weight: 0.01, encoding: "exact" },
    object:      { query: text.slice(0, 500), weight: 1.0, encoding: "semantic" },
  }, 5);

  // 4. Risk flags
  const riskFlags: string[] = [];
  const lower = text.toLowerCase();
  if (lower.includes("in no one event")) riskFlags.push("⚠ Likely drafting error: 'IN NO ONE EVENT' (should be 'IN NO EVENT')");
  if (lower.includes("perpetual") && lower.includes("irrevocable")) riskFlags.push("⚠ Perpetual + irrevocable license detected — high IP risk");
  if (lower.includes("unlimited") || (lower.includes("no limit") && lower.includes("liab"))) riskFlags.push("⚠ Uncapped liability language detected");
  if (/\b(90|ninety)\s*day/.test(lower) && lower.includes("terminat")) riskFlags.push("⚠ 90-day termination notice — market norm is 30 days");

  return {
    hbPrediction,
    hbScore,
    vanillaPrediction,
    vanillaScore,
    allScores,
    rescued: vanillaPrediction !== hbPrediction,
    peerClauses: peerHits.slice(1), // exclude the top hit (itself)
    riskFlags,
  };
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ScoreBar({ label, value, max = 1, color }: { label: string; value: number; max?: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="w-48 truncate font-mono text-xs text-gray-400">{label}</span>
      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-12 text-right font-mono text-xs" style={{ color }}>
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function BenchmarkTable() {
  return (
    <div className="overflow-hidden rounded-xl border border-white/10">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/10 text-left">
            <th className="px-4 py-3 text-gray-400 font-medium">Model</th>
            <th className="px-4 py-3 text-gray-400 font-medium text-right">P@1</th>
            <th className="px-4 py-3 text-gray-400 font-medium">Training</th>
            <th className="px-4 py-3 text-gray-400 font-medium">Compute</th>
          </tr>
        </thead>
        <tbody>
          {BENCHMARK_DATA.map((row, i) => (
            <tr
              key={i}
              className={`border-b border-white/5 last:border-0 transition-colors ${
                row.highlight ? "bg-emerald-500/10" : "hover:bg-white/3"
              }`}
            >
              <td className="px-4 py-3">
                <span className={row.highlight ? "text-emerald-400 font-semibold" : "text-gray-300"}>
                  {row.model}
                </span>
              </td>
              <td className="px-4 py-3 text-right font-mono">
                <span className={row.highlight ? "text-emerald-400 font-bold" : "text-gray-300"}>
                  {row.p1}%
                </span>
              </td>
              <td className="px-4 py-3 text-gray-400">{row.training}</td>
              <td className="px-4 py-3 text-gray-400">{row.compute}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AblationChart() {
  return (
    <div className="space-y-3">
      {ABLATION_DATA.map((row) => {
        const gain = row.multiSlot - row.vanilla;
        return (
          <div key={row.type} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-gray-400 font-mono">{row.type}</span>
              <span className="text-emerald-400 font-mono font-bold">+{gain.toFixed(1)}pp</span>
            </div>
            <div className="relative h-5 bg-white/5 rounded overflow-hidden">
              {/* Vanilla bar */}
              <div
                className="absolute inset-y-0 left-0 bg-red-500/40 rounded"
                style={{ width: `${row.vanilla}%` }}
              />
              {/* HyperBinder bar */}
              <div
                className="absolute inset-y-0 left-0 bg-emerald-500/60 rounded"
                style={{ width: `${row.multiSlot}%` }}
              />
              <div className="absolute inset-0 flex items-center justify-between px-2">
                <span className="text-[10px] font-mono text-white/70">HB {row.multiSlot}%</span>
                <span className="text-[10px] font-mono text-white/50">Vanilla {row.vanilla}%</span>
              </div>
            </div>
          </div>
        );
      })}
      <div className="flex gap-4 pt-2 text-xs text-gray-500">
        <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded bg-emerald-500/60" />HyperBinder</span>
        <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded bg-red-500/40" />Vanilla k-NN</span>
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

type Tab = "classify" | "benchmark" | "ablation";

export default function Page() {
  const [tab, setTab] = useState<Tab>("classify");
  const [clauseText, setClauseText] = useState("");
  const [selectedSample, setSelectedSample] = useState<SampleClause | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClassify = useCallback(async () => {
    if (!clauseText.trim() || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await runClassify(clauseText);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Classification failed");
    } finally {
      setLoading(false);
    }
  }, [clauseText, loading]);

  const tabs: { id: Tab; label: string }[] = [
    { id: "classify",  label: "Live Classifier" },
    { id: "benchmark", label: "Benchmark" },
    { id: "ablation",  label: "Ablation" },
  ];

  return (
    <main className="min-h-screen bg-[#080c10] text-white selection:bg-emerald-500/30">
      {/* ── Noise texture overlay ── */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
          backgroundSize: "128px",
        }}
      />

      {/* ── Grid lines ── */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.04]"
        style={{
          backgroundImage:
            "linear-gradient(to right, #10b981 1px, transparent 1px), linear-gradient(to bottom, #10b981 1px, transparent 1px)",
          backgroundSize: "80px 80px",
        }}
      />

      <div className="relative max-w-5xl mx-auto px-6 py-16">

        {/* ── Hero ── */}
        <div className="mb-14">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-xs font-mono mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Live · HyperBinder Engine · CUAD 510 contracts indexed
          </div>

          <h1
            className="text-5xl font-bold tracking-tight leading-none mb-4"
            style={{ fontFamily: "var(--font-geist-sans)" }}
          >
            <span className="text-white">Beating Fine-Tuned</span>
            <br />
            <span className="text-emerald-400">Transformers.</span>
            <span className="text-gray-600"> Without Training.</span>
          </h1>

          <p className="text-gray-400 text-lg max-w-2xl leading-relaxed">
            94.8% P@1 on CUAD-SL. No GPU. No fine-tuning. Outperforms DeBERTa by{" "}
            <span className="text-white font-semibold">7 percentage points</span> using
            hyperdimensional symbolic retrieval.
          </p>

          {/* Stat pills */}
          <div className="flex flex-wrap gap-3 mt-8">
            {[
              ["94.8%", "P@1 Accuracy"],
              ["1,538", "Clauses evaluated"],
              ["510", "Contracts indexed"],
              ["0", "Parameters trained"],
              ["7.4pp", "Gain over vanilla k-NN"],
            ].map(([val, lbl]) => (
              <div
                key={lbl}
                className="flex flex-col px-4 py-2.5 rounded-lg border border-white/8 bg-white/3"
              >
                <span className="text-xl font-bold font-mono text-emerald-400">{val}</span>
                <span className="text-xs text-gray-500 mt-0.5">{lbl}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Tabs ── */}
        <div className="flex gap-1 p-1 rounded-xl bg-white/5 border border-white/8 mb-8 w-fit">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-5 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                tab === t.id
                  ? "bg-emerald-500 text-black shadow-lg shadow-emerald-500/20"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* ══════════════════════════════════════════════════════════════════
            TAB: LIVE CLASSIFIER
        ══════════════════════════════════════════════════════════════════ */}
        {tab === "classify" && (
          <div className="space-y-6">

            {/* Input card */}
            <div className="rounded-2xl border border-white/10 bg-white/3 p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-widest">
                  Clause Input
                </h2>
                <div className="flex gap-2 flex-wrap justify-end">
                  {Object.entries(SAMPLE_CLAUSES).map(([label, sample]) => (
                    <button
                      key={label}
                      onClick={() => {
                        setClauseText(sample.text);
                        setSelectedSample(sample);
                        setResult(null);
                      }}
                      className="text-xs px-2.5 py-1 rounded-md border border-white/10 text-gray-400 hover:text-emerald-400 hover:border-emerald-500/40 transition-colors font-mono"
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Ground truth banner — shown when a known sample is loaded */}
              {selectedSample && !result && (
                <div className={`rounded-xl border px-4 py-3 text-sm font-mono space-y-1 ${
                  selectedSample.vanillaWrong
                    ? "border-blue-500/30 bg-blue-500/10"
                    : "border-emerald-500/30 bg-emerald-500/10"
                }`}>
                  {selectedSample.vanillaWrong ? (
                    <>
                      <div className="text-blue-400 font-semibold">📋 Known rescued clause — ground truth available</div>
                      <div className="text-gray-400">
                        True type: <span className="text-white">{selectedSample.trueType}</span>
                        &nbsp;·&nbsp;
                        Vanilla k-NN gets: <span className="text-red-400">{selectedSample.vanillaWrong}</span>
                        &nbsp;·&nbsp;
                        Source: <span className="text-gray-500">{selectedSample.contract}</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="text-emerald-400 font-semibold">✓ Both systems should agree on this one</div>
                      <div className="text-gray-400">
                        True type: <span className="text-white">{selectedSample.trueType}</span>
                        &nbsp;·&nbsp;
                        <span className="text-gray-500">Unambiguous clause — watch both systems classify correctly</span>
                      </div>
                    </>
                  )}
                </div>
              )}

              <textarea
                value={clauseText}
                onChange={(e) => { setClauseText(e.target.value); setResult(null); setSelectedSample(null); }}
                placeholder="Paste a contract clause here, or use a sample above…"
                rows={6}
                className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-600 resize-none focus:outline-none focus:border-emerald-500/50 font-mono leading-relaxed"
              />

              <button
                onClick={handleClassify}
                disabled={!clauseText.trim() || loading}
                className="w-full py-3 rounded-xl font-semibold text-sm transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed bg-emerald-500 text-black hover:bg-emerald-400 active:scale-[0.99] shadow-lg shadow-emerald-500/20"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                    </svg>
                    Running 10 parallel queries…
                  </span>
                ) : (
                  "Classify Clause →"
                )}
              </button>
            </div>

            {/* Error */}
            {error && (
              <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-red-400 text-sm font-mono">
                ✗ {error}
              </div>
            )}

            {/* Results */}
            {result && (
              <div className="space-y-4 animate-in fade-in duration-500">

                {/* Head-to-head verdict */}
                <div className="grid grid-cols-2 gap-4">
                  {/* HyperBinder */}
                  <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-5">
                    <div className="text-xs font-mono text-emerald-500 mb-2 uppercase tracking-widest">HyperBinder</div>
                    <div className="text-xl font-bold text-white mb-1">{result.hbPrediction}</div>
                    <div className="text-sm font-mono text-emerald-400">
                      {(result.hbScore * 100).toFixed(2)}% confidence
                    </div>
                    <div className="mt-3 text-xs text-gray-500">Symbolic filter · 10 parallel subspace queries</div>
                  </div>

                  {/* Vanilla */}
                  <div className={`rounded-2xl border p-5 ${
                    result.rescued
                      ? "border-red-500/30 bg-red-500/5"
                      : "border-white/10 bg-white/3"
                  }`}>
                    <div className="text-xs font-mono text-gray-500 mb-2 uppercase tracking-widest">Vanilla k-NN</div>
                    <div className={`text-xl font-bold mb-1 ${result.rescued ? "text-red-400" : "text-white"}`}>
                      {result.vanillaPrediction}
                    </div>
                    <div className="text-sm font-mono text-gray-400">
                      {(result.vanillaScore * 100).toFixed(2)}% confidence
                    </div>
                    <div className="mt-3 text-xs text-gray-500">Global pool · single query</div>
                  </div>
                </div>

                {/* Agreement banner — both correct */}
                {!result.rescued && selectedSample && !selectedSample.vanillaWrong && (
                  <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-emerald-400 text-sm font-mono flex items-center gap-2">
                    <span>✓</span>
                    <span>
                      <strong>Both systems agree.</strong> This is an unambiguous clause — vanilla k-NN and HyperBinder both classify it correctly.
                    </span>
                  </div>
                )}

                {/* Rescued banner */}
                {result.rescued && (
                  <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-amber-400 text-sm font-mono space-y-1">
                    <div className="flex items-center gap-2">
                      <span>🎯</span>
                      <span>
                        <strong>Rescued clause.</strong> Vanilla classified as{" "}
                        <span className="underline text-red-400">{result.vanillaPrediction}</span> — symbolic filter
                        corrected to <span className="underline text-emerald-400">{result.hbPrediction}</span>.
                      </span>
                    </div>
                    {selectedSample && result.hbPrediction === selectedSample.trueType && (
                      <div className="text-emerald-400 text-xs">
                        ✓ Confirmed by ground truth — true label is {selectedSample.trueType}
                      </div>
                    )}
                  </div>
                )}

                {/* Risk flags */}
                {result.riskFlags.length > 0 && (
                  <div className="rounded-2xl border border-orange-500/30 bg-orange-500/5 p-5 space-y-2">
                    <div className="text-xs font-mono text-orange-400 uppercase tracking-widest mb-3">Risk Flags</div>
                    {result.riskFlags.map((flag, i) => (
                      <div key={i} className="text-sm text-orange-300 font-mono">{flag}</div>
                    ))}
                  </div>
                )}

                {/* All scores */}
                <div className="rounded-2xl border border-white/10 bg-white/3 p-5 space-y-2.5">
                  <div className="text-xs font-mono text-gray-500 uppercase tracking-widest mb-4">
                    Score breakdown — all 10 clause types
                  </div>
                  {Object.entries(result.allScores)
                    .sort((a, b) => b[1] - a[1])
                    .map(([ct, score]) => (
                      <ScoreBar
                        key={ct}
                        label={ct}
                        value={score}
                        max={Math.max(...Object.values(result.allScores))}
                        color={ct === result.hbPrediction ? "#10b981" : "#374151"}
                      />
                    ))}
                </div>

                {/* Peer clauses */}
                {result.peerClauses.length > 0 && (
                  <div className="rounded-2xl border border-white/10 bg-white/3 p-5 space-y-4">
                    <div className="text-xs font-mono text-gray-500 uppercase tracking-widest">
                      Peer clauses — why this classification
                    </div>
                    {result.peerClauses.slice(0, 3).map((hit, i) => (
                      <div key={i} className="border-l-2 border-emerald-500/30 pl-4 space-y-1">
                        <div className="flex gap-3 items-center">
                          <span className="text-xs font-mono text-emerald-400">
                            {(hit._score * 100).toFixed(1)}% match
                          </span>
                          {hit.data.contract && (
                            <span className="text-xs text-gray-600 font-mono truncate">
                              {hit.data.contract}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-gray-400 leading-relaxed line-clamp-3">
                          {hit.data.object ?? "—"}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            TAB: BENCHMARK
        ══════════════════════════════════════════════════════════════════ */}
        {tab === "benchmark" && (
          <div className="space-y-6">
            <div className="rounded-2xl border border-white/10 bg-white/3 p-6">
              <h2 className="text-sm font-mono text-gray-500 uppercase tracking-widest mb-1">
                Classification accuracy vs published benchmarks
              </h2>
              <p className="text-gray-400 text-sm mb-6">
                CUAD-SL · 1,538 clause instances · 510 contracts · O'Connell et al. (2025)
              </p>
              <BenchmarkTable />
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/3 p-6">
              <h2 className="text-sm font-mono text-gray-500 uppercase tracking-widest mb-4">
                Per-clause-type breakdown
              </h2>
              <div className="overflow-hidden rounded-xl border border-white/10">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/10 text-left">
                      <th className="px-4 py-3 text-gray-400 font-medium">Clause Type</th>
                      <th className="px-4 py-3 text-gray-400 font-medium text-right">N</th>
                      <th className="px-4 py-3 text-gray-400 font-medium text-right">P@1</th>
                      <th className="px-4 py-3 text-gray-400 font-medium text-right">P@3</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { type: "Anti-Assignment",                 n: 374, p1: 98.7, p3: 99.2 },
                      { type: "Non-Compete",                     n: 119, p1: 98.3, p3: 99.2 },
                      { type: "Irrevocable Or Perpetual License",n: 70,  p1: 98.6, p3: 100  },
                      { type: "Ip Ownership Assignment",         n: 124, p1: 97.6, p3: 99.2 },
                      { type: "Uncapped Liability",              n: 111, p1: 96.4, p3: 100  },
                      { type: "Termination For Convenience",     n: 183, p1: 97.3, p3: 98.4 },
                      { type: "Covenant Not To Sue",             n: 100, p1: 97.0, p3: 98.0 },
                      { type: "Liquidated Damages",              n: 61,  p1: 95.1, p3: 100  },
                      { type: "Change Of Control",               n: 121, p1: 90.1, p3: 99.2 },
                      { type: "Cap On Liability",                n: 275, p1: 78.9, p3: 99.6 },
                    ].map((row, i) => (
                      <tr key={i} className="border-b border-white/5 last:border-0 hover:bg-white/3 transition-colors">
                        <td className="px-4 py-2.5 text-gray-300 font-mono text-xs">{row.type}</td>
                        <td className="px-4 py-2.5 text-gray-500 text-right font-mono text-xs">{row.n}</td>
                        <td className="px-4 py-2.5 text-right font-mono text-xs">
                          <span className={row.p1 < 85 ? "text-amber-400" : "text-emerald-400"}>
                            {row.p1}%
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-400">{row.p3}%</td>
                      </tr>
                    ))}
                    <tr className="bg-emerald-500/5 border-t border-emerald-500/20">
                      <td className="px-4 py-3 text-emerald-400 font-semibold font-mono text-xs">Macro Average</td>
                      <td className="px-4 py-3 text-emerald-400 text-right font-mono text-xs">1,538</td>
                      <td className="px-4 py-3 text-emerald-400 text-right font-mono text-xs font-bold">94.8%</td>
                      <td className="px-4 py-3 text-emerald-400 text-right font-mono text-xs">99.3%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            TAB: ABLATION
        ══════════════════════════════════════════════════════════════════ */}
        {tab === "ablation" && (
          <div className="space-y-6">
            <div className="rounded-2xl border border-white/10 bg-white/3 p-6">
              <h2 className="text-sm font-mono text-gray-500 uppercase tracking-widest mb-1">
                Symbolic filter ablation
              </h2>
              <p className="text-gray-400 text-sm mb-6">
                HyperBinder multi-slot vs vanilla k-NN — same Legal-BERT embeddings, same index.
                The only variable is the symbolic clause_type filter.
              </p>

              <div className="grid grid-cols-3 gap-4 mb-8">
                {[
                  ["7.4pp", "Net accuracy gain"],
                  ["160",   "Clauses rescued"],
                  ["46",    "Clauses lost"],
                ].map(([val, lbl]) => (
                  <div key={lbl} className="rounded-xl border border-white/8 bg-white/3 p-4 text-center">
                    <div className="text-2xl font-bold font-mono text-emerald-400">{val}</div>
                    <div className="text-xs text-gray-500 mt-1">{lbl}</div>
                  </div>
                ))}
              </div>

              <AblationChart />
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/3 p-6">
              <h2 className="text-sm font-mono text-gray-500 uppercase tracking-widest mb-3">
                Why the filter matters
              </h2>
              <div className="space-y-3 text-sm text-gray-400 leading-relaxed">
                <p>
                  Standard k-NN operates over a <span className="text-white">single global embedding space</span> where all clause
                  types compete simultaneously. Frequent types like{" "}
                  <span className="text-amber-400 font-mono">Anti-Assignment (374 instances)</span> exert
                  gravitational pull on every query, regardless of true similarity.
                </p>
                <p>
                  HyperBinder's multi-slot architecture issues{" "}
                  <span className="text-emerald-400 font-mono">one query per candidate clause type</span>,
                  each evaluated within only that type's subspace. The winning prediction is the type
                  whose intra-class similarity is highest — not the type that dominates the global pool.
                </p>
                <p>
                  The most dramatic rescue:{" "}
                  <span className="text-white font-semibold">Irrevocable Or Perpetual License</span> drops
                  from 98.6% to 52.9% without the filter — because License Grant clauses flood the
                  candidate pool.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ── Footer ── */}
        <div className="mt-20 pt-8 border-t border-white/5 flex items-center justify-between text-xs text-gray-600 font-mono">
          <span>HyperBinder · Semantic Reach · Legal AI · 2026</span>
          <span>CUAD · Hendrycks et al. NeurIPS 2021</span>
        </div>
      </div>
    </main>
  );
}