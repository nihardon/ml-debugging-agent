import React, { useRef, useState } from "react";

const ACCEPT_MAP = {
  csv_file: ".csv",
  config_file: ".yaml,.yml,.json",
  log_file: ".txt,.log",
};

const LABELS = {
  csv_file: "Loss Curve CSV",
  config_file: "Config (YAML / JSON)",
  log_file: "Training Log (.txt)",
};

function FileDropZone({ field, file, onChange }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) onChange(field, dropped);
  };

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-4 cursor-pointer transition-colors
        ${dragging ? "border-indigo-400 bg-indigo-900/20" : "border-gray-600 hover:border-indigo-500 bg-gray-800/40"}`}
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_MAP[field]}
        className="hidden"
        onChange={(e) => e.target.files[0] && onChange(field, e.target.files[0])}
      />
      <div className="flex items-center gap-3">
        <span className="text-2xl select-none">
          {field === "csv_file" ? "📈" : field === "config_file" ? "⚙️" : "📄"}
        </span>
        <div className="min-w-0">
          <p className="text-sm font-medium text-gray-300">{LABELS[field]}</p>
          {file ? (
            <p className="text-xs text-indigo-400 truncate">{file.name}</p>
          ) : (
            <p className="text-xs text-gray-500">Drop file or click to browse</p>
          )}
        </div>
        {file && (
          <button
            className="ml-auto text-gray-500 hover:text-red-400 text-xs"
            onClick={(e) => { e.stopPropagation(); onChange(field, null); }}
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}

export default function UploadPanel({ onResult, onLossData, onToken, onStreamStart }) {
  const [files, setFiles] = useState({ csv_file: null, config_file: null, log_file: null });
  const [stackTrace, setStackTrace] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (field, file) => {
    setFiles((prev) => ({ ...prev, [field]: file }));

    // If a CSV is added, immediately parse it for the chart
    if (field === "csv_file" && file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        // Normalise Windows \r\n and strip trailing whitespace
        const text = e.target.result.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
        const lines = text.split("\n").filter((l) => l.trim() !== "");
        if (lines.length < 2) { onLossData([]); return; }

        // Strip surrounding quotes from each cell (handles Excel-exported CSVs)
        const parseCell = (cell) => cell.trim().replace(/^["']|["']$/g, "");

        const headers = lines[0].split(",").map(parseCell).map((h) => h.toLowerCase());

        // Match by exact name first, then partial match
        const STEP_NAMES = ["step", "epoch", "iter", "iteration", "steps", "epochs"];
        const LOSS_NAMES = ["loss", "train_loss", "training_loss", "value"];
        const VAL_NAMES  = ["val_loss", "validation_loss", "valid_loss", "eval_loss", "test_loss"];

        let stepIdx    = headers.findIndex((h) => STEP_NAMES.includes(h));
        let lossIdx    = headers.findIndex((h) => LOSS_NAMES.includes(h));
        let valLossIdx = headers.findIndex((h) => VAL_NAMES.includes(h));

        // Partial match fallbacks
        if (stepIdx    < 0) stepIdx    = headers.findIndex((h) => STEP_NAMES.some((n) => h.includes(n)));
        if (lossIdx    < 0) lossIdx    = headers.findIndex((h) => LOSS_NAMES.some((n) => h.includes(n)));
        if (valLossIdx < 0) valLossIdx = headers.findIndex((h) => VAL_NAMES.some((n) => h.includes(n)));

        // Last resort: column 0 = step, column 1 = loss
        const si = stepIdx >= 0 ? stepIdx : 0;
        const li = lossIdx >= 0 ? lossIdx : 1;

        const points = lines.slice(1).map((line) => {
          const cols = line.split(",").map(parseCell);
          const pt = { step: parseFloat(cols[si]), loss: parseFloat(cols[li]) };
          if (valLossIdx >= 0) {
            const vl = parseFloat(cols[valLossIdx]);
            if (!isNaN(vl)) pt.val_loss = vl;
          }
          return pt;
        }).filter((p) => !isNaN(p.step) && !isNaN(p.loss));

        onLossData(points.length > 0 ? points : []);
      };
      reader.readAsText(file);
    }
    if (field === "csv_file" && !file) {
      onLossData(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    const hasFile = Object.values(files).some(Boolean);
    const hasTrace = stackTrace.trim().length > 0;
    if (!hasFile && !hasTrace) {
      setError("Please upload at least one file or paste a stack trace.");
      return;
    }

    const formData = new FormData();
    if (files.log_file) formData.append("log_file", files.log_file);
    if (files.csv_file) formData.append("csv_file", files.csv_file);
    if (files.config_file) formData.append("config_file", files.config_file);
    if (stackTrace.trim()) formData.append("stack_trace", stackTrace.trim());

    setLoading(true);
    onStreamStart?.();

    try {
      const res = await fetch("/diagnose/stream", { method: "POST", body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        // SSE events are separated by double newlines
        const parts = buffer.split("\n\n");
        buffer = parts.pop(); // keep any incomplete trailing chunk

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;
          let payload;
          try {
            payload = JSON.parse(line.slice(6));
          } catch {
            continue;
          }

          if (payload.type === "token") {
            onToken?.(payload.content);
          } else if (payload.type === "done") {
            onResult(payload.report);
          } else if (payload.type === "error") {
            throw new Error(payload.message);
          }
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 border border-gray-700">
      <h2 className="text-lg font-semibold text-white mb-5 flex items-center gap-2">
        <span className="text-indigo-400">⬆</span> Upload Training Artifacts
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* File drop zones */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {Object.keys(files).map((field) => (
            <FileDropZone
              key={field}
              field={field}
              file={files[field]}
              onChange={handleFileChange}
            />
          ))}
        </div>

        {/* Stack trace paste area */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-1">
            Stack Trace (paste here)
          </label>
          <textarea
            className="w-full h-36 bg-gray-800 border border-gray-600 rounded-lg p-3 text-sm
                       font-mono text-gray-300 focus:outline-none focus:border-indigo-500
                       resize-none placeholder-gray-600"
            placeholder={"Traceback (most recent call last):\n  ...\nRuntimeError: CUDA out of memory."}
            value={stackTrace}
            onChange={(e) => setStackTrace(e.target.value)}
          />
        </div>

        {error && (
          <div className="rounded-lg bg-red-900/40 border border-red-700 px-4 py-2 text-sm text-red-300">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 rounded-lg font-semibold text-sm tracking-wide transition-all
                     bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 disabled:cursor-not-allowed
                     text-white flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Diagnosing…
            </>
          ) : (
            "Run Diagnosis"
          )}
        </button>
      </form>
    </div>
  );
}
