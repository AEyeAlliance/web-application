from flask import Flask, render_template, request
import sys

sys.path.insert(0, 'final_model')
from space_recognition_original import make_prediction

app = Flask(__name__)

from flask_uploads import UploadSet, configure_uploads, IMAGES
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static'
configure_uploads(app, photos)

@app.route("/")
def index():
    return render_template('home.html')
 
@app.route("/aboutus")
def aboutus():
    return render_template('aboutus.html')

@app.route("/tutorial_slider")
def tutorial_slider():
    return render_template('tutorial_slider.html')

@app.route("/upload")
def upload():
    return render_template('upload.html')

@app.route("/upload_results", methods=['GET', 'POST'])
def save():
    
    # Save the image in the path
    if request.method == 'POST' and 'fileField' in request.files:
        filename = photos.save(request.files['fileField'])
    
    img_path = "static/" + filename
    predicted_letter = make_prediction(img_path)
    return render_template('display.html', filename=filename, letter=predicted_letter)

if __name__ == "__main__":
    app.run(debug=True)
