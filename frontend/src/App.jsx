import React, { useEffect, useRef, useState } from "react";
import DiagnosticReport from "./components/DiagnosticReport";
import LossChart from "./components/LossChart";
import UploadPanel from "./components/UploadPanel";

function KBStatus() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch("/health")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus({ status: "error" }));
  }, []);

  if (!status) return null;

  const seeded = status.kb_seeded;
  return (
    <div
      className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border
        ${seeded
          ? "bg-green-900/30 border-green-700 text-green-400"
          : "bg-yellow-900/30 border-yellow-700 text-yellow-400"
        }`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${seeded ? "bg-green-400" : "bg-yellow-400"}`} />
      KB: {status.kb_doc_count ?? 0} docs {!seeded && "— run seed_kb.py"}
    </div>
  );
}

function StreamingPanel({ text }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [text]);

  return (
    <div className="bg-gray-900 rounded-xl border border-indigo-700/50 h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2.5 px-4 py-3 border-b border-gray-800">
        <span className="relative flex h-2.5 w-2.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-indigo-500" />
        </span>
        <span className="text-sm font-medium text-indigo-300">Claude is thinking…</span>
        <span className="ml-auto text-xs text-gray-600 tabular-nums">{text.length} chars</span>
      </div>

      {/* Streaming text */}
      <pre
        className="flex-1 overflow-y-auto p-4 text-xs text-gray-400 font-mono whitespace-pre-wrap
                   leading-relaxed scrollbar-thin scrollbar-thumb-gray-700"
      >
        {text}
        <span className="inline-block w-1.5 h-3.5 bg-indigo-400 ml-0.5 align-text-bottom animate-pulse" />
      </pre>
    </div>
  );
}

export default function App() {
  const [report, setReport] = useState(null);
  const [lossData, setLossData] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");

  const handleStreamStart = () => {
    setReport(null);
    setStreamingText("");
    setIsStreaming(true);
  };

  const handleToken = (chunk) => {
    setStreamingText((prev) => prev + chunk);
  };

  const handleResult = (data) => {
    setReport(data);
    setIsStreaming(false);
    setStreamingText("");
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Nav */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xl">🤖</span>
            <span className="font-semibold text-white tracking-tight">ML Debugging Agent</span>
          </div>
          <KBStatus />
        </div>
      </header>

      {/* Main layout */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column: Upload + Chart */}
          <div className="space-y-6">
            <UploadPanel
              onResult={handleResult}
              onLossData={setLossData}
              onToken={handleToken}
              onStreamStart={handleStreamStart}
            />
            <LossChart
              lossData={lossData}
              divergenceStep={report?.divergence_step ?? null}
            />
          </div>

          {/* Right column: Streaming panel → Report → Empty state */}
          <div className="min-h-64">
            {isStreaming ? (
              <StreamingPanel text={streamingText} />
            ) : report ? (
              <DiagnosticReport report={report} />
            ) : (
              <div className="bg-gray-900 rounded-xl border border-gray-700 h-full min-h-64
                              flex flex-col items-center justify-center gap-3 p-8 text-center">
                <span className="text-5xl opacity-30">🩺</span>
                <p className="text-gray-500 text-sm max-w-xs">
                  Upload your training artifacts and click <strong className="text-gray-400">Run Diagnosis</strong> to
                  get a structured debugging report.
                </p>
                <ul className="text-xs text-gray-600 space-y-1 mt-2">
                  <li>• Loss curve CSV</li>
                  <li>• Training log .txt</li>
                  <li>• Config YAML / JSON</li>
                  <li>• Stack trace (paste)</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
