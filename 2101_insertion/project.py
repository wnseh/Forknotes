import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from predict import predict
app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
ALLOWED_EXTENSIONS = {'png'}

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')

def process_file(path, filename):
    #predict
    filename = 'downloaded_text.txt'
    full_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    file1 = open(full_path, 'w')
    predicted_text = 'ssup' 
    #TODO predicted_text = predict(full_path)
    file1.write(predicted_text)
    file1.close

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'],filename, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
