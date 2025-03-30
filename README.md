# Vashisht2.0
# Problem Statement
Develop a deep learning-based and NLP system that detects, classifies, and auto-completes stuttered speech in real-time, improving accessibility and communication.
# Team Members:
Raghavan R ,
Derrick Richard ,
Tarun Srikumar ,
Sreenidhi K

# Solution Overview:
This ReactJS application leverages deep learning and NLP models to assist individuals with stuttering. It comprises three core modules:

## Stutter Detection & Classification:

Utilizes a pre-trained deep learning model (Wave2vec) to analyze audio input and classify different types of stutters (e.g., repetitions, prolongations, and blocks).

Provides real-time feedback to users with visual and audio indicators.

## Stutter Correction/Aide with BERT:

Incorporates a fine-tuned BERT model to suggest corrections and alternatives for detected stuttered speech.

Generates context-aware suggestions to guide users in improving speech fluency.

## Speech Therapy Component:

Interactive speech therapy exercises with configuration options to modify pacing according to the user's stutter profile.

# Running the App:
cd Frontend/FLOWSpeak && npm install && npm run dev

# Running the Backend:
cd Backend && pip install -r requirements.txt && python setup.py && python server.py

# For the Demo Video Click Below:
<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=LUXPLMexKSI
" target="_blank"><img src="http://img.youtube.com/vi/LUXPLMexKSI/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="500" height="400" border="10" /></a></p>

# For the Setup Click Below:
<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=mrcyOX2VKk4
" target="_blank"><img src="http://img.youtube.com/vi/mrcyOX2VKk4/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="500" height="400" border="10" /></a></p>

## Real World Applications
### Upon implementing Text To Speech, you may implement the stutter correction component to be used in online meetings for people with stuttering (block) problems , to automatically aide them
### Practice speaking using speech therapy component before any important speeches for people with stuttering (repetition) problems
### We would also love to take inputs from people with ideas on how to help people who have stuttering issues through the contact form.
