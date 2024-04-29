import cv2 
from hand_tracker import Tracker
from color_gradient import get_color_gradient
import matplotlib.pyplot as plt
import numpy as np
import text2emotion as te

from matplotlib.animation import FuncAnimation

from scamp import *


# Initialize camera parameters
width = 1280
height = 720
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
tracker = Tracker()

# Fixing random state for reproducibility
np.random.seed(19680801)

# Get user input and emotion
text = input("Describe your feelings today: ")
moods = te.get_emotion(text)

# Desgined mood color codes
mood_colors = {'Happy' : '#FAD619', 
               'Angry' : '#CC1111',
               'Surprise' : '#5C4AB3',
               'Sad' : '#5F8EC5',
               'Fear' : '#1C655C'}

# Pre-selected instruments
mood_instruments = {'Surprise' : ['brass', 'Steel Drum'],
                    'Fear' : ['Viola LP2','Space Voice'],
                    'Happy' : ['Music Box', 'Bird'],
                    'Sad' : ['Whistle', 'Flute Gold'],
                    'Angry' : ['Overdrive Guitar','Taiko Drum']
                    }

# Tuned note offset for moods
mood_offsets = {'Surprise' : -2,
                    'Fear' : 10,
                    'Happy' : 5,
                    'Sad' : -5,
                    'Angry' : -5
                }

# Create new figure and axes
fig = plt.figure(figsize=(7, 7), facecolor='black')
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])

# Create graphics data
n_drops = 20
rain_drops = np.zeros(n_drops, dtype=[('position', float, (2,)),
                                      ('size',     float),
                                      ('growth',   float),
                                      ('color',    float, (4,))])

# Initialize pattern parameters
x1, y1 = (0.5, 0.5)
alpha = 0.85  # transparency of patterns
expansion = 0.1  # random scatter of patterns around x1, y1

# Initialize patterns in random positions and with random growth rates
rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
rain_drops['growth'] = np.random.uniform(50, 200, n_drops)

# Initialize color as the mood color with most probability
max_mood = max(moods, key=moods.get)
max_mood_color = mood_colors[max_mood]
rain_drops['color'][:, 0:3] = get_color_gradient("#FFFFFF", max_mood_color, n_drops)
rain_drops['color'][:, 3] = alpha

# Initialize music session with a main instrument and a secondary instrument 
# corresponding to the mood
s = Session().run_as_server()
main_inst = s.new_part(mood_instruments[max_mood][0])
sec_inst = s.new_part(mood_instruments[max_mood][1])

# For each mood, assign a percetage of the markers to be the preset color
# that corresponds to the detected probability of the emotion in the user input text
for mood, percent in moods.items():
    print(mood, percent)
    if percent > 0 and mood != max_mood:
        rand_percent = np.random.uniform(0, 1, n_drops)
        colors = get_color_gradient("#FFFFFF", mood_colors[mood], n_drops)
        for j in range(n_drops):
            if (rand_percent[j] < percent):
                rain_drops['color'][j, 0:3] = colors[j]

# Construct the scatter which we will update during animation
scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1], 
                  s=rain_drops['size'], lw=0.5, 
                  c=rain_drops['color'],
                  marker='8' )


thres = 50  # distance threshold
prev_pos = np.array([x1, y1])  # position of index finger from previous frame
counter = 0  # input string counter
sec_prob = 0.05  # probability of secondary instrument playing a note

def update(frame_number):
    global prev_pos
    global counter

    _, img = cap.read()
    img = cv2.flip(img, 1)
    img = tracker.hand_landmark(img)
    img, x1, y1 = tracker.tracking(img) # x1, y1 = tracked location of the index finger

    # Play a note only when distance between the previous position and current position
    # of the index finger is greater than a distance threshold
    if np.linalg.norm(prev_pos - np.array([x1, y1])) > thres:
        # Get a char from user input (looped)
        char = text[counter%len(text)]
        if char.isalnum():
            # Main instrument plays a note with mood offset
            main_inst.play_note(ord(char) - 20 + mood_offsets[max_mood], 1, 1)
            # Secondary instrument plays a note under a set probability
            if (np.random.uniform(0, 1) < 0.2):
                sec_inst.play_note(ord(char) - 20, 1, 1)

    img = cv2.resize(img, (width//3, height//3))
    # Uncomment lines below to show webcam stream
    # cv2.imshow('Image', img)
    # cv2.waitKey(1)

    # Get an index to re-spawn the oldest pattern
    current_index = frame_number % n_drops

    # Make all colors more transparent as time progresses
    rain_drops['color'][:, 3] -= 1.0/len(rain_drops)
    rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)

    # Make all circles bigger.
    rain_drops['size'] += rain_drops['growth']

    # Pick a new position for oldest marker, resetting its size and growth factor.
    # Position is normalized index finger position (x, y) with a value of 'expansion' random oscillation
    new_pos = np.array([x1/width, 1-y1/height]) + np.random.uniform(-expansion, expansion, 2) 
    rain_drops['position'][current_index] = np.clip(new_pos, 0, 1)
    rain_drops['size'][current_index] = 1
    rain_drops['growth'][current_index] = np.random.uniform(50, 150)

    # Update the scatter collection, with the new sizes and positions.
    scat.set_sizes(rain_drops['size'])
    scat.set_offsets(rain_drops['position'])

    # Update position and counter
    prev_pos = np.array([x1, y1])
    counter += 1



# Construct the animation, using the update function as the animation director.
animation = FuncAnimation(fig, update, interval=10, save_count=100)
plt.show()