# AI-Based Motion Tracking and Rehabilitation for Wrist and Finger Therapy

## Description

This project provides **real-time, AI-powered tracking and analysis** of wrist and finger motion for rehabilitation purposes. It uses **MediaPipe** and **computer vision** to quantify wrist flexion/extension, thumb opposition, finger PIP/DIP angles, and other clinically relevant metrics during therapy sessions. **Data is logged at each step** for subsequent analysis. After each session, users receive **Excel spreadsheets and detailed PDF reports** summarizing their progress and metrics.

---

## Features

- **Live video feed** with wrist and finger landmark detection and annotation[1][2].
- **Real-time calculation** of wrist flexion, extension, radial/ulnar deviation, and finger joint angles (PIP, DIP, CMC, MCP, IP)[2].
- **Gesture identification**: fist clench, thumb opposition, and open hand states are recognized[2].
- **Session logging** with automatic generation of Excel sheets, plots PDF, and summary PDF for each session[1][2].
- **Web interface** for session management, real-time feedback, and report download[1][3].
- **Clinically relevant feedback** with encouragement messages based on exercise performance[2].

---

## How It Works

1. **User starts a session** via the web interface, optionally naming it for future reference[1][3].
2. **Hand is tracked in real time** using the device camera. MediaPipe analyzes joint angles, gestures, and provides live visual feedback in the browser[2].
3. **Metrics are logged every frame**: wrist and finger angles, gesture types, and timing data[2].
4. **At the end of each session**, reports are generated:
    - **Excel** file with time-series of all angles and gestures[2].
    - **Plots PDF** showing wrist/finger motion over time[2].
    - **Summary PDF** with mean, min, max, and clinical normal ranges for each metric[2].
5. **Users can review and download** previous sessions’ reports directly from the web interface[1][3].

---

## Technical Stack

- **Backend:** Flask (Python)[1]
- **Computer Vision:** MediaPipe, OpenCV[2]
- **Data Analysis:** Pandas, NumPy, Matplotlib[2]
- **Frontend:** HTML with minimal JavaScript for video feeds and asynchronous updates[1][3]
- **Data Storage:** Local filesystem (each session saved in `outputs/` subdirectory)[1][2]

---

## Installation

1. **Clone the repository**:
   <pre>bash git clone https://github.com/your-username/AI-Based-Motion-Tracking-and-Rehabilitation-for-Wrist-and-Finger-Therapy.git
   cd AI-Based-Motion-Tracking-and-Rehabilitation-for-Wrist-and-Finger-Therapy </pre>
   
2. **(Optional) Create a virtual environment**:
    ```python -m venv venv```
    ```source venv/bin/activate```
    ```# On Windows: venv\Scripts\activate```

3. **Install dependencies**:
    ```pip install -r requirements.txt```

4. **Run the application**:
    ```python app.py```

5. **Visit `http://127.0.0.1:5000` in your browser to start.**

---

## Usage

- **Start a session**: Enter a session name (or leave blank for auto-naming) and click "Start"[1][3].
- **Perform exercises**: Follow the on-screen guidance and video feedback[2][4].
- **End session**: Click "End" to stop tracking and generate reports[1][4].
- **Download reports**: Access Excel, plots PDF, and summary PDF via the results or logs page[1][3][5].
- **Review history**: All previous sessions are listed and downloadable from the main page[1][3].

---

## Outputs

- **Excel (.xlsx):** Time-series data for all tracked metrics and gestures[2].
- **Plots PDF:** Visualizations of angles and distances over the session duration[2].
- **Summary PDF:** Tabular summary with mean, min, max, and clinical normal ranges for each angle[2].

---

## Supported Exercises (Examples)

- **Wrist flexion/extension**
- **Radial/ulnar deviation**
- **Thumb opposition** (touching each fingertip)
- **Finger flexion/extension** (PIP, DIP joints)
- **Fist clench**

---

## Acknowledgments

Special thanks to **MediaPipe**, **OpenCV**, and **Matplotlib** for enabling real-time hand tracking and visualization.  
This project is designed for clinical rehabilitation, providing **accessible, quantitative feedback** for hand therapy.
