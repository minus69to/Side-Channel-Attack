function app() {
  return {
    latencyResults: null,
    traceData: [],
    heatmaps: [],
    traceStats: [],
    status: "",
    isCollecting: false,
    statusIsError: false,
    showingTraces: false,

    // Task 1: Collect latency data
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        let worker = new Worker("warmup.js");
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });
        this.latencyResults = results;
        this.status = "Latency data collection complete!";
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Task 2: Collect Trace
    async collectTraceData() {
      this.isCollecting = true;
      this.status = "Collecting sweep trace data...";
      this.statusIsError = false;
      try {
        let worker = new Worker("worker.js");
        const sweepTrace = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });
        worker.terminate();

        const response = await fetch("/collect_trace", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ trace: sweepTrace }),
        });

        if (!response.ok) throw new Error("Backend error during trace upload.");
        const result = await response.json();
        if (!result.success) throw new Error("Trace upload failed.");

        this.traceData.push(sweepTrace);
        if (result.heatmap) this.heatmaps.push(result.heatmap);
        if (result.stats) this.traceStats.push(result.stats);

        this.status = "Trace data collected & heatmap generated!";
        this.showingTraces = true;
      } catch (error) {
        console.error("Error collecting trace data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // ==== ADDED: Download Traces ====
    async downloadTraces() {
      this.status = "Downloading trace dataset...";
      this.statusIsError = false;
      try {
        const response = await fetch("/api/get_results");
        if (!response.ok) throw new Error("Failed to fetch traces.");
        const data = await response.json();
        const blob = new Blob([JSON.stringify(data.traces, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "traces.json";
        a.click();
        URL.revokeObjectURL(url);
        this.status = "Download started.";
      } catch (e) {
        this.status = "Failed to download traces.";
        this.statusIsError = true;
      }
    },

    // ==== ADDED: Refresh Results ====
    async fetchResults() {
      this.status = "Refreshing traces and heatmaps...";
      this.statusIsError = false;
      try {
        const response = await fetch("/api/get_results");
        if (!response.ok) throw new Error("Failed to fetch results.");
        const data = await response.json();
        // Populate local arrays if backend provides heatmaps/stats
        if (data.traces) this.traceData = data.traces;
        // Uncomment and adapt below if your backend returns heatmaps/stats:
        // if (data.heatmaps) this.heatmaps = data.heatmaps;
        // if (data.stats) this.traceStats = data.stats;
        this.status = "Results refreshed!";
      } catch (e) {
        this.status = "Failed to refresh results.";
        this.statusIsError = true;
      }
    },

    // ==== ADDED: Clear All Results ====
    async clearResults() {
      this.status = "Clearing all results...";
      this.statusIsError = false;
      try {
        const response = await fetch("/api/clear_results", { method: "POST" });
        if (!response.ok) throw new Error("Failed to clear results.");
        this.latencyResults = null;
        this.traceData = [];
        this.heatmaps = [];
        this.traceStats = [];
        this.showingTraces = false;
        this.status = "Cleared";
      } catch (e) {
        this.status = "Failed to clear results.";
        this.statusIsError = true;
      }
    },
  };
}
window.app = app;
