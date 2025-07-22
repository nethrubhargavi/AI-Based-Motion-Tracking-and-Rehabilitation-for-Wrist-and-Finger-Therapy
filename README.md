# AI-Based Motion Tracking and Rehabilitation for Wrist and Finger Therapy

## Project Description

This application uses **Flask**, **OpenCV**, and **MediaPipe** to provide real-time, AI-driven hand tracking for wrist and finger rehabilitation. It captures video from your webcam, analyzes hand and finger movements, and provides quantitative feedback on joint angles, flexion/extension, finger opposition, and fist clenching—essential metrics for physical therapy. After each session, it generates downloadable reports (Excel, PDF) for patients and clinicians.

## Features

- **Real-time hand & finger tracking** using MediaPipe’s advanced pose estimation[2].
- **Quantitative metrics**: Wrist flexion/extension, radial/ulnar deviation, thumb and finger joint angles (CMC, MCP, IP, PIP, DIP), thumb-finger opposition distances, and fist clench metrics[2].
- **Automated feedback**: Encouragement and form prompts based on movement quality[2].
- **Session recording**: Every session is stored with detailed time-series data and downloadable reports (Excel, plots PDF, summary PDF)[1][2].
- **Web interface**: Accessible via browser for both patients and clinicians[1][3].
- **Privacy**: All data is stored locally; no patient information is transmitted.

## How It Works

- **Start** a session via the web interface (optionally naming it)[1].
- **Perform** therapy exercises while the app tracks joint angles and provides feedback in real time[2].
- **End** the session to automatically generate downloadable reports showing detailed analytics and summary statistics[1][2].
- **View session history** and download previous reports from the home screen[1][3].

## Tech Stack

- **Backend**: Python, Flask[1]
- **Computer Vision**: OpenCV, MediaPipe[2]
- **Data Analysis**: pandas, Matplotlib[2]
- **Frontend**: HTML, basic Jinja2 templating[1][3]
