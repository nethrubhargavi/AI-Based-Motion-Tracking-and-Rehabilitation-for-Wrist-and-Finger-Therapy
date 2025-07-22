# AI-Based-Motion-Tracking-and-Rehabilitation-for-Wrist-and-Finger-Therapy

This repository contains a Flask-based web application that leverages MediaPipe for real-time hand tracking to assist with wrist and finger rehabilitation therapy. The system tracks joint angles, wrist flexion/extension, and finger opposition, providing automated feedback and detailed session analytics for patients and clinicians.


#✨ **Features**
1. Real-time hand and finger tracking: Uses webcam input to monitor wrist and finger movements during exercises.

2. Comprehensive metrics: Measures wrist flexion/extension, finger joint angles (PIP, DIP, CMC, MCP), thumb opposition distances, and fist clench dynamics.

3. Automated feedback: Provides real-time encouragement and corrective prompts based on movement quality.

4. Session recording & analytics: Saves each therapy session with detailed time-series data, plots, and summary statistics (Excel, PDF).

5. Responsive web interface: Accessible via browser with session history and downloadable reports.

6. Configurable for clinical use: Normal joint ranges and validation rules are integrated for wrist and finger rehabilitation protocols.


Tech Stack
Backend: Python, Flask

Computer Vision: OpenCV, MediaPipe

Data Analysis & Visualization: pandas, Matplotlib

Frontend: HTML, Jinja2 (lightweight templating)

