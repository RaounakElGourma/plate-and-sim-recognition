import os
import re
import io
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import cv2
import numpy as np
import easyocr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

app = Flask(__name__)
app.secret_key = 'secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

reader = easyocr.Reader(['fr', 'en'], gpu=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# Corrected regex: raw string, proper escapes
PHONE_REGEX = re.compile(r'(?:\+212|0)?(?:6|7)\d{8}')
PHONE_CLEAN = re.compile(r'[^0-9+]')

def process_image(filepath, out_filename=None):
    img = cv2.imread(filepath)
    if img is None:
        return []

    results = reader.readtext(img)
    detections = []
    for bbox, text, conf in results:
        # Normalize common OCR confusions
        cleaned = (
            text.strip()
                .replace('O', '0')
                .replace('o', '0')
                .replace('l', '1')
                .replace('I', '1')
        )
        digits = PHONE_CLEAN.sub('', cleaned)
        found = PHONE_REGEX.findall(digits)
        if found:
            for f in found:
                detections.append({
                    'bbox': bbox,
                    'text': f,
                    'raw': text,
                    'conf': float(conf)
                })

    if out_filename:
        vis = img.copy()
        for d in detections:
            pts = np.array(d['bbox'], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            x, y = pts[0]
            cv2.putText(vis, d['text'], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(out_filename, vis)

    return detections

RESULT_ROWS = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            flash('No file selected')
            return redirect(request.url)

        RESULT_ROWS.clear()
        total, detected = 0, 0

        for f in files:
            if f and allowed_file(f.filename):
                total += 1
                ext = f.filename.rsplit('.', 1)[1].lower()
                uid = uuid.uuid4().hex
                filename = f'{uid}.{ext}'
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(save_path)

                out_name = f'proc_{filename}'
                out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)

                detections = process_image(save_path, out_path)
                if detections:
                    detected += 1

                # Fallback to original image if annotated one wasn't created
                if not os.path.exists(out_path):
                    src = cv2.imread(save_path)
                    if src is not None:
                        cv2.imwrite(out_path, src)

                row = {
                    'id': uid,
                    'image': filename,
                    'processed_image': out_name,
                    'detections': ';'.join([d['text'] for d in detections]),
                    'confidences': ';'.join([str(d['conf']) for d in detections])
                }
                RESULT_ROWS.append(row)

        rate = f"{detected}/{total} ({(detected/total*100):.2f}%)" if total else '0/0 (0%)'

        # Fresh figure for each request and round pie
        fig_path = os.path.join(app.config['PROCESSED_FOLDER'], 'stats.png')
        plt.figure()
        plt.pie([detected, total - detected], labels=['Detected', 'Not detected'], autopct='%1.1f%%')
        plt.title('Detection success rate')
        plt.axis('equal')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        return render_template('results.html',
                               rows=RESULT_ROWS,
                               stats_image='processed/stats.png',
                               rate=rate)

    return render_template('index.html')

@app.route('/download/<ftype>')
def download(ftype):
    if not RESULT_ROWS:
        flash('No results to export')
        return redirect(url_for('index'))

    df = pd.DataFrame(RESULT_ROWS)

    if ftype == 'csv':
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return send_file(io.BytesIO(buf.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='results.csv')

    elif ftype == 'xlsx':
        buf = io.BytesIO()
        # Requires openpyxl or xlsxwriter installed
        df.to_excel(buf, index=False)
        buf.seek(0)
        return send_file(buf,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True,
                         download_name='results.xlsx')

    elif ftype == 'pdf':
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        y = 750
        for row in RESULT_ROWS:
            line = f"{row['image']} | {row['detections']} | {row['confidences']}"
            c.drawString(50, y, line[:100])
            y -= 20
            if y < 50:
                c.showPage()
                y = 750
        c.save()
        buf.seek(0)
        return send_file(buf,
                         mimetype='application/pdf',
                         as_attachment=True,
                         download_name='results.pdf')

    return redirect(url_for('index'))

@app.route('/edit', methods=['POST'])
def edit():
    data = request.get_json(silent=True) or {}
    for row in RESULT_ROWS:
        if row['id'] == data.get('id'):
            row['detections'] = data.get('value', '')
            return jsonify({'status': 'ok'})
    return jsonify({'status': 'not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
