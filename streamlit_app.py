import streamlit as st
import cv2
import time
from handtrackingmodule import HandDetector

def main():
    st.title("Hand Gesture Game 🎮🖐️")
    
    run = st.checkbox('Start Webcam')

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    detector = HandDetector()

    while run:
        success, frame = camera.read()
        frame = detector.findHands(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()

if __name__ == '__main__':
    main()
