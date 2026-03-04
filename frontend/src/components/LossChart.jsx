import React, { useMemo } from "react";
import Plot from "react-plotly.js";

export default function LossChart({ lossData, divergenceStep }) {
  const { xVals, yVals } = useMemo(() => {
    if (!lossData || lossData.length === 0) return { xVals: [], yVals: [] };
    return {
      xVals: lossData.map((p) => p.step),
      yVals: lossData.map((p) => p.loss),
    };
  }, [lossData]);

  if (!lossData) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-700 flex items-center justify-center h-64">
        <p className="text-gray-500 text-sm">Upload a CSV loss curve to visualize it here.</p>
      </div>
    );
  }

  if (lossData.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-700 flex items-center justify-center h-64">
        <p className="text-yellow-500 text-sm">
          CSV uploaded but no numeric data could be parsed. Check that your file has{" "}
          <code className="font-mono bg-gray-800 px-1 rounded">step</code> and{" "}
          <code className="font-mono bg-gray-800 px-1 rounded">loss</code> columns.
        </p>
      </div>
    );
  }

  const traces = [
    {
      x: xVals,
      y: yVals,
      type: "scatter",
      mode: "lines",
      name: "Training Loss",
      line: { color: "#818cf8", width: 2 },
      hovertemplate: "Step %{x}<br>Loss: %{y:.4f}<extra></extra>",
    },
  ];

  const shapes = [];
  const annotations = [];

  if (divergenceStep != null) {
    shapes.push({
      type: "line",
      x0: divergenceStep,
      x1: divergenceStep,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: "#f87171", width: 2, dash: "dash" },
    });
    annotations.push({
      x: divergenceStep,
      y: 1,
      yref: "paper",
      text: "Divergence Detected",
      showarrow: true,
      arrowhead: 2,
      arrowcolor: "#f87171",
      font: { color: "#f87171", size: 12 },
      bgcolor: "rgba(30,10,10,0.7)",
      bordercolor: "#f87171",
      borderwidth: 1,
      borderpad: 4,
      ax: 40,
      ay: -30,
    });
  }

  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-700">
      <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
        <span className="text-indigo-400">📉</span> Loss Curve
        {divergenceStep != null && (
          <span className="ml-auto text-xs font-normal text-red-400 bg-red-900/30 border border-red-700 rounded px-2 py-0.5">
            Divergence @ step {divergenceStep}
          </span>
        )}
      </h2>
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "#111827",
          font: { color: "#d1d5db", family: "monospace, sans-serif" },
          xaxis: {
            title: { text: "Step / Epoch", font: { size: 12 } },
            gridcolor: "#374151",
            linecolor: "#374151",
            zeroline: false,
          },
          yaxis: {
            title: { text: "Loss", font: { size: 12 } },
            gridcolor: "#374151",
            linecolor: "#374151",
            zeroline: false,
          },
          margin: { t: 10, r: 20, b: 50, l: 55 },
          shapes,
          annotations,
          showlegend: false,
          autosize: true,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "280px" }}
        useResizeHandler
      />
    </div>
  );
}
