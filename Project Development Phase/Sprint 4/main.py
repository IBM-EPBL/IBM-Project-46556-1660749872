from flask import Flask,render_template,Response
import cv2
from tensorflow.keras.models import load_model  # to load our trained model
import numpy as np
from werkzeug.utils import secure_filename


app=Flask(__name__)
camera=cv2.VideoCapture(0)
model = load_model("disaster.h5")



def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame1 = cv2.resize(frame1, (64, 64))
        x = np.expand_dims(frame1, axis=0)
        result = np.argmax(model.predict(x), axis=-1)
        index = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
        result = str(index[result[0]])
        frame = cv2.putText(frame,'The Predicted is: '+str(result),(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        print(result)
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    global result
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)