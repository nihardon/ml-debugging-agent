import React, { useState } from "react";
import Plot from "react-plotly.js";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------
const STATUS_CONFIG = {
  Critical: { bg: "bg-red-900/50", border: "border-red-600", text: "text-red-300", dot: "bg-red-500" },
  Warning:  { bg: "bg-yellow-900/40", border: "border-yellow-600", text: "text-yellow-300", dot: "bg-yellow-400" },
  Healthy:  { bg: "bg-green-900/40", border: "border-green-600", text: "text-green-300", dot: "bg-green-400" },
};

function StatusBadge({ status }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.Warning;
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold border ${cfg.bg} ${cfg.border} ${cfg.text}`}>
      <span className={`w-2 h-2 rounded-full ${cfg.dot}`} />
      {status}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Confidence gauge
// ---------------------------------------------------------------------------
function ConfidenceGauge({ value }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.7 ? "#4ade80" : value >= 0.4 ? "#facc15" : "#f87171";

  return (
    <div className="flex items-center gap-3">
      <Plot
        data={[
          {
            type: "indicator",
            mode: "gauge+number",
            value: pct,
            number: { suffix: "%", font: { size: 18, color: "#e5e7eb" } },
            gauge: {
              axis: { range: [0, 100], tickwidth: 0, tickcolor: "transparent", tickfont: { color: "transparent" } },
              bar: { color, thickness: 0.7 },
              bgcolor: "#1f2937",
              borderwidth: 0,
              steps: [{ range: [0, 100], color: "#374151" }],
            },
          },
        ]}
        layout={{
          width: 120,
          height: 80,
          margin: { t: 0, b: 0, l: 10, r: 10 },
          paper_bgcolor: "transparent",
          font: { color: "#e5e7eb" },
        }}
        config={{ displayModeBar: false, staticPlot: true }}
      />
      <div>
        <p className="text-xs text-gray-500">Confidence</p>
        <p className="text-sm font-semibold" style={{ color }}>{pct}%</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section wrapper
// ---------------------------------------------------------------------------
function Section({ title, icon, children }) {
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 flex items-center gap-1.5">
        <span>{icon}</span>{title}
      </h3>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Copyable code block
// ---------------------------------------------------------------------------
function CopyableCode({ code }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="relative rounded-lg overflow-hidden border border-gray-700 text-sm group">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 z-10 px-2.5 py-1 rounded text-xs font-medium
                   bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white
                   opacity-0 group-hover:opacity-100 transition-all duration-150"
      >
        {copied ? "✓ Copied" : "Copy"}
      </button>
      <SyntaxHighlighter
        language="python"
        style={vscDarkPlus}
        customStyle={{ margin: 0, borderRadius: 0, background: "#0d1117" }}
        showLineNumbers
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function DiagnosticReport({ report }) {
  if (!report) return null;

  const {
    status,
    root_cause,
    confidence,
    explanation,
    fix_code_snippet,
    ranked_actions,
    what_to_monitor,
    citations,
    divergence_step,
  } = report;

  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `diagnosis-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-700 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="text-xl">🩺</span>
          <h2 className="text-lg font-semibold text-white">Diagnostic Report</h2>
          <StatusBadge status={status} />
        </div>
        <div className="flex items-center gap-3">
          <ConfidenceGauge value={confidence} />
          <button
            onClick={handleDownload}
            className="px-3 py-1.5 rounded-lg text-xs font-medium border border-gray-600
                       text-gray-400 hover:text-white hover:border-gray-400 transition-colors"
            title="Download report as JSON"
          >
            ↓ JSON
          </button>
        </div>
      </div>

      <div className="px-6 py-5 space-y-6">
        {/* Root cause + explanation */}
        <Section title="Root Cause" icon="🔍">
          <p className="text-white font-medium">{root_cause}</p>
          <p className="text-gray-400 text-sm leading-relaxed">{explanation}</p>
          {divergence_step != null && (
            <p className="text-xs text-red-400 mt-1">
              Divergence detected at step <strong>{divergence_step}</strong>
            </p>
          )}
        </Section>

        {/* Fix code snippet */}
        {fix_code_snippet && (
          <Section title="Recommended Fix" icon="🔧">
            <CopyableCode code={fix_code_snippet} />
          </Section>
        )}

        {/* Ranked actions */}
        {ranked_actions && ranked_actions.length > 0 && (
          <Section title="Action Plan" icon="📋">
            <ol className="space-y-2">
              {ranked_actions
                .slice()
                .sort((a, b) => a.priority - b.priority)
                .map((item, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-800 text-indigo-200
                                     text-xs font-bold flex items-center justify-center mt-0.5">
                      {item.priority}
                    </span>
                    <span className="text-sm text-gray-300">{item.action}</span>
                  </li>
                ))}
            </ol>
          </Section>
        )}

        {/* What to monitor */}
        {what_to_monitor && (
          <Section title="What to Monitor" icon="📡">
            <p className="text-sm text-gray-300 bg-gray-800 rounded-lg px-4 py-2.5 border border-gray-700">
              {what_to_monitor}
            </p>
          </Section>
        )}

        {/* Citations */}
        {citations && citations.length > 0 && (
          <Section title="Citations" icon="📚">
            <div className="flex flex-wrap gap-2">
              {citations.map((c, i) => (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs
                             bg-gray-800 border border-gray-600 text-gray-400"
                  title={c.source}
                >
                  <span className="text-indigo-400 font-mono">[{c.id}]</span>
                  {c.source}
                </span>
              ))}
            </div>
          </Section>
        )}
      </div>
    </div>
  );
}
