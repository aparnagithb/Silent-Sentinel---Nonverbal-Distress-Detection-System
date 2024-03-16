<h1 align="center" style="border-bottom: none">
    <b>
        <a href="https://www.google.com"> Silent Sentinel: Technology that Empowers, When Words Fail </a><br>
    </b>
    ⭐️Taking Back Control, One Signal at a Time  ⭐️ <br>
</h1>

 [`[Demo video link](https://youtu.be/t9vbOHfHoAM) `]
Silent Guardian: A lifesaver for the voiceless. Uses cameras to detect silent cries for help through sign language, sending immediate alerts for those in danger.
## Team Details
`Team number` : VH204

| Name                  | Email                    |
|-----------------------|--------------------------|
| M.Aparna Lakshmi      | mapa22079.cs@rmkec.ac.in |
| Kavi Varshini .S      | kavi22177.cs@rmkec.ac.in |
| Jebisha Anish Mary.A  | Jebi22054.cs@rmkec.ac.in |
| kavya Sree .M.S       | kavy22066.cs@rmkec.ac.in |



## Problem statement 
Many individuals facing emergencies, including victims of domestic violence, people with hearing or speech impairments, and those in high-stress situations, are unable to call for help verbally. This project aims to develop a real-time sign language detection system that can identify pre-defined emergency signs from various sign languages (including Indian Sign Language and universal signs) using video surveillance cameras. Upon detecting an emergency sign, the system will automatically send an SOS alert to designated responders, such as local police stations, to ensure timely intervention and protect vulnerable individuals.
## About the project
Silent Guardian is a real-time sign language detection system designed to enhance public safety and support individuals in need.

Project Overview:
- Uses Computer vision for hand and sign detection.
- Machine learning models for gender classification , emotion and sign Estimation(custom dataset).
- Real-time video processing.
- Alert system for designated responders (local police station, ambulances, fire departments, and medical facilities ).

Target Scenarios:
- Public places with pre-existing surveillance cameras -deaf institutions, jails, elevators, buses, supermarkets, orphanages, medical shops, local shops,schools, universities, airports, train stations, and government buildings, etc.
- Private CCTV systems with user consent

  ![workflow](https://github.com/aparnagithb/Silent-Sentinel---Nonverbal-Distress-Detection-System/assets/119504238/b4d91f17-6919-4c5a-85cb-074e7bf44024)



## Technical implemntaion 

 ![WhatsApp Image 2024-03-16 at 21 26 49_c38c0417](https://github.com/aparnagithb/Silent-Sentinel---Nonverbal-Distress-Detection-System/assets/119504238/a0c98407-3c8c-4e32-a06e-2acba0d5280a)

Data Collection and Preprocessing:
- Developed a custom dataset of emergency signs with 10 signs of over 2000 images totally in Indian Sign Language , hand symbols issued at Canadian Woman Foundation for times of distress and other relevant sign languages.
- Utilized existing datasets for gender and emotion recognition from Kaggle and Github.
- Preprocessed all data by resizing (maintaining aspect ratio), cropping the region of interest (ROI), and converting to grayscale.

Model Training:
Train separate machine learning models for:
- Hand detection using pre-trained models or custom approaches.
- Sign classification using my custom dataset and pre-trained models (Teachable Machine developed by google - uses MobileNet) for fine-tuning.
- Gender and emotion recognition models 

Real-time Video Processing:
- Develop a system to continuously capture video frames from surveillance cameras.
- Implement hand and sign detection algorithms to identify raised hands and potential emergency signs.
- Classify the detected signs using the trained sign classification model.

Alert System and User Interface:
- Upon detecting an emergency sign with a high confidence score(calculated by using the signs detected along with the emotional status of the person so that when someone accidently signs it doesn't generate a false signal ), trigger an automated SOS alert.
- Send the alert to designated responders (e.g., local police station) with relevant information (camera location, time, potential emergency type based on the sign).
- Develop a user interface (dashboard) to visualize camera feeds, detected signs, and any triggered alerts.
## Techstacks used 
![image](https://github.com/aparnagithb/Silent-Sentinel---Nonverbal-Distress-Detection-System/assets/119504238/164c4b5f-06ad-4ce8-8159-e4622216eb77)1.OpenCV, cvzone
 ![image](https://github.com/aparnagithb/Silent-Sentinel---Nonverbal-Distress-Detection-System/assets/119504238/f69cfa70-4ddf-43e4-872b-186bc03a4380)2.NumPy
![image](https://github.com/aparnagithb/Silent-Sentinel---Nonverbal-Distress-Detection-System/assets/119504238/60b64e0b-1349-4997-a0dc-72f1d0b42374)3.TensorFlow/Keras
![image](https://github.com/aparnagithb/Silent-Sentinel---Nonverbal-Distress-Detection-System/assets/119504238/167e2f3b-3cca-4aff-a510-442f7c66b350)4.MediaPipe 

5.Teachable Machine (by google)

## How to run locally 
1.Create a virtual environment to isolate project dependencies and avoid conflicts. Tools like venv or conda can be used.
2.pip install tensorflow==2.16.1 cvzone==1.6.1 keras==3.0.5 mediapipe==0.10.1     #installing newer versions of these libraries could cause compatability issues while compiling
3.Clone my project repository from the remote location using the provided URL or command.
4.Use the cd command in your terminal to navigate to the directory where your project's code resides
5.Execute the main script using Python: python test.py



# What's next ?
Our Silent Guardian project has a lot of potential to expand its capabilities and impact in the future.
- Audio Cues: Incorporate audio processing to detect cries for help or suspicious sounds alongside visual cues.
- Multi-Sign Recognition: Develop the system to recognize sequences of signs for more complex messages
- Multi-Camera Network: Develop a system that can handle video feeds from multiple cameras across different locations, enabling centralized monitoring.
- Contextual Awareness: Implement machine learning algorithms to analyze contextual information, such as the location, time of day, and repeating user behavior patterns, to provide more accurate results .
- - Enhance the system to cater to different sign languages such as American Sign Language (ASL), French Sign Language (LSF), and Indian Sign Language (ISL)

## Declaration
We confirm that the project showcased here was either developed entirely during the hackathon or underwent significant updates within the hackathon timeframe. We understand that if any plagiarism from online sources is detected, our project will be disqualified, and our participation in the hackathon will be revoked.
