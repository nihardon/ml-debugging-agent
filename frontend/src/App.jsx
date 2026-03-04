import React, { useEffect, useState } from "react";
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

export default function App() {
  const [report, setReport] = useState(null);
  const [lossData, setLossData] = useState(null);

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
            <UploadPanel onResult={setReport} onLossData={setLossData} />
            <LossChart
              lossData={lossData}
              divergenceStep={report?.divergence_step ?? null}
            />
          </div>

          {/* Right column: Report */}
          <div>
            {report ? (
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
