from flask import Flask, render_template, Response, redirect, url_for, session, send_from_directory, request, jsonify
from final_code import HandTracker
import os
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)
tracker = HandTracker()

def list_sessions():
    sessions = []
    if not os.path.exists('outputs'):
        return sessions
    for session_dir in sorted(os.listdir('outputs'), reverse=True):
        folder = os.path.join('outputs', session_dir)
        if os.path.isdir(folder):
            sessions.append({
                'name': session_dir,
                'excel': next((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.xlsx')), None),
                'plots_pdf': next((os.path.join(folder, f) for f in os.listdir(folder) if 'plots' in f and f.endswith('.pdf')), None),
                'summary_pdf': next((os.path.join(folder, f) for f in os.listdir(folder) if 'summary' in f and f.endswith('.pdf')), None),
            })
    return sessions

@app.route('/')
def index():
    session.clear()
    sessions = list_sessions()
    return render_template('index.html', sessions=sessions)

@app.route('/start', methods=['POST'])
def start():
    session_name = request.form.get('session_name')
    if not session_name:
        session_name = f"session_{int(time.time())}"
    session['user_session_name'] = session_name
    return redirect(url_for('live'))

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video_feed_raw')
def video_feed_raw():
    return Response(tracker.raw_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_processed')
def video_feed_processed():
    return Response(tracker.processed_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/feedback')
def get_feedback():
    feedback_message = tracker.get_feedback()
    return jsonify({'message': feedback_message})

@app.route('/end', methods=['POST'])
def end():
    session_name = session.get('user_session_name', f"session_{int(time.time())}")
    outputs = tracker.stop_tracking(session_name=session_name)
    session['outputs'] = outputs
    session['current_session'] = session_name
    return redirect(url_for('results'))

@app.route('/results')
def results():
    outputs = session.get('outputs', {})
    session_name = session.get('current_session', 'Unknown')
    return render_template('results.html', outputs=outputs, session_name=session_name)

@app.route('/download/<filetype>/<session_name>')
def download(filetype, session_name):
    folder = os.path.join('outputs', session_name)
    file_map = {
        'excel': next((f for f in os.listdir(folder) if f.endswith('.xlsx')), None),
        'plots_pdf': next((f for f in os.listdir(folder) if 'plots' in f and f.endswith('.pdf')), None),
        'summary_pdf': next((f for f in os.listdir(folder) if 'summary' in f and f.endswith('.pdf')), None),
    }
    filename = file_map.get(filetype)
    if filename:
        return send_from_directory(folder, filename, as_attachment=True)
    return "File not found", 404

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    app.run(debug=True)
