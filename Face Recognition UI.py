import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import os
import face_recognition
import numpy as np

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.textvalue = tkinter.StringVar()

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
                
        self.label = tkinter.Label(window,text = "Enter your name here")
        self.label.pack()

        self.userentry = tkinter.Entry(window, textvariable = self.textvalue,width = 100)
        self.userentry.pack()                
        
        self.btn_train=tkinter.Button(window, text="Train", width=50, command=self.train)
        self.btn_train.pack(anchor=tkinter.CENTER, expand=True)
        
        self.btn_recognition=tkinter.Button(window, text="Face Recognize", width=50, command=self.facerecognition)
        self.btn_recognition.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()
        
    def train(self):
        # Load a sample picture and learn how to recognize it.
        os.chdir('database')
        data = os.listdir()
        lst = [os.path.splitext(x)[0] for x in data]        
        self.known_face_encodings = []
        self.known_face_names = []
                
                
        for name in lst:
            variable =  str(name)+"_image"
            variable2 = str(name)+"_encodings"
            variable = face_recognition.load_image_file(str(name)+".jpg")
            variable2= face_recognition.face_encodings(variable)[0]
            self.known_face_encodings.append(variable2)
            self.known_face_names.append(name)
        
        os.chdir("..")
        
    def facerecognition(self):
        self.vid = MyVideoCapture(self.video_source)
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        
        while True:
            # Grab a single frame of video
            ret, frame = self.vid.get_frame()
        
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
        
            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
        
                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]
        
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
        
                    face_names.append(name)
        
            process_this_frame = not process_this_frame
        
        
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
        
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
            # Display the resulting image
            cv2.imshow('Video', frame)
        
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.vid.release()
        cv2.destroyAllWindows() 
    
    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        
        if ret:
            os.chdir('database')
            image = str(self.textvalue.get()) +".jpg"
            cv2.imwrite(image,cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            os.chdir('..')
        return image

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            # Create a window and pass it to the Application object


App(tkinter.Tk(), "Face Recognition")