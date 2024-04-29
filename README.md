# Moodwave

This repository contains the final project by Dina Long for the course 10335 Art and Machine Learning. 

Moodwave is an interactive system that combines user mood input with music and visual elements to reflect their emotional states. Our core approach leverages Machine Learning recognition and tracking of hand movements and users’ moods to create a dynamic experience that adjusts based on user inputs and environmental factors.

Techniques are as follows:
1. Text to Mood
    - Prompt the user to answer the question: “Describe your feelings today”
    - Use text2emotion, a Python package that utilizes natural language processing techniques, to process the text and categorize the emotions embedded in it into 5 different emotion categories: Happy, Angry, Sad, Surprise and Fear.

2. Mood Graphics
    - Describe to Chatgpt our goal of this project, and ask for recommendations of colors that represent different emotions.
    
3. Mood Music 
    - Use SCAMP (Suite for Computer-Assisted Music in Python)
        - Provides a list of instrument presets
        - Plays notes with given note number, velocity, and length
    - Map each emotion to a main instrument and a secondary instrument
    - Map each character in the user input text to a music note

4. Hand Tracking & User Interaction
    - Use Python’s OpenCV library and the Hand Landmarker from Mediapipe to track the location of the user’s index finger in real time from camera input. 
    - Use Matplotlib to generate and animate graphic patterns
    - Match the movements of the visual graphic patterns with users’ hand movements. 
    - Speed of hand movement controls the speed of the music 
