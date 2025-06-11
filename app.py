# ==== FIX: Prevent matplotlib Tkinter runtime error ====
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend

from flask import Flask, send_from_directory, request, jsonify
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []
stored_stats = []  # ==== TASK 2: Store stats per trace ====

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

# ==== TASK 2: Trace Collection Endpoint with Min/Max/Range/Sample Count ====
@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    data = request.get_json()
    trace = data.get('trace', [])
    stored_traces.append(trace)

    arr = np.array(trace)
    stats = {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "range": int(arr.max() - arr.min()),
        "samples": int(arr.size)
    }
    stored_stats.append(stats)

    # Generate heatmap
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.imshow([trace], aspect='auto', cmap='viridis')
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Sweep Count')
    ax.set_yticks([]) 
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    heatmap_url = "data:image/png;base64," + image_base64
    stored_heatmaps.append(heatmap_url)

    # ==== Return stats for frontend ====
    return jsonify({'success': True, 'heatmap': heatmap_url, 'stats': stats})

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    stored_traces.clear()
    stored_heatmaps.clear()
    stored_stats.clear()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
