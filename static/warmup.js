/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  // 1. Allocate a buffer of n * LINESIZE bytes
  let buffer = new Uint8Array(n * LINESIZE);
  let timings = [];

  // 2. Repeat 10 times to get a stable median measurement
  for (let repeat = 0; repeat < 10; repeat++) {
    let t0 = performance.now();

    // 3. Read first byte of every cache line
    for (let i = 0; i < n; i++) {
      let tmp = buffer[i * LINESIZE];
    }

    let t1 = performance.now();
    timings.push(t1 - t0);
  }

  // 4. Sort timings and take the median value
  timings.sort((a, b) => a - b);
  let mid = Math.floor(timings.length / 2);
  let median;
  if (timings.length % 2 === 0) {
    median = (timings[mid - 1] + timings[mid]) / 2;
  } else {
    median = timings[mid];
  }
  return median;
}

// Worker event listener for main thread messages
self.addEventListener("message", function (e) {
  // === CHANGED: only handle "start" message ===
  if (e.data === "start") {
    const results = {};
    // Values of n: from 1 to 10,000,000 in steps (as in spec)
    // You can adjust the range if browser crashes or memory runs out
    const nValues = [
      1, 10, 100, 1000, 10000, 100000, 1000000, 10000000
    ];
    for (let n of nValues) {
      try {
        results[n] = readNlines(n);
      } catch (e) {
        // If memory allocation fails or other error, break the loop
        break;
      }
    }
    self.postMessage(results);
  }
});
