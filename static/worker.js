const LINESIZE = 64; // bytes
const LLCSIZE = 16 * 1024 * 1024; // 16 MB = 16 * 1024 * 1024 bytes
const TIME = 10000;
const P = 10; 

function sweep(P) {
  const numLines = Math.floor(LLCSIZE / LINESIZE);
  let buffer = new Uint8Array(numLines * LINESIZE);
  let K = Math.floor(TIME / P);
  let sweepCounts = new Array(K).fill(0);

  for (let k = 0; k < K; k++) {
    let start = performance.now();
    let count = 0;
    while (performance.now() - start < P) {
      for (let i = 0; i < numLines; i++) {
        let temp = buffer[i * LINESIZE];
      }
      count++;
    }
    sweepCounts[k] = count;
  }
  return sweepCounts;
}

self.addEventListener('message', function(e) {
  if (e.data === "start") {
    let trace = sweep(P);
    self.postMessage(trace);
  }
});
