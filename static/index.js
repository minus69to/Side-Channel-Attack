function app() {
  return {
    latencyResults: null,
    traceData: [],
    heatmaps: [],
    traceStats: [], // ==== TASK 2: Store stats per trace ====
    status: "",
    isCollecting: false,
    statusIsError: false,
    showingTraces: false,

    // Task 1: Unchanged
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

    // ==== TASK 2: Collect Trace, Show Heatmap and Stats ====
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

        // Send to backend
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

    // Placeholders for other buttons if needed
    async downloadTraces() {},
    async clearResults() {},
  };
}
window.app = app;
