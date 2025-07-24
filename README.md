# Smart Glasses for the Visually Impaired

**A real time assistive technology prototype to help blind individuals read text and detect objects using smart glasses powered by Raspberry Pi.**

![Project Demo](prototype.jpg)



---

## Description

This prototype smart glasses system enables visually impaired users to:

- **Read printed or handwritten text**
- **Detect objects in their path**
- **Receive audio feedback in realtime**

The project is designed to be lightweight and runs on **Raspberry Pi 4**, making it portable and affordable for real-world use cases.



---

## Core Features

### Object Detection
- **Script:** `obj_detection.py`
- Uses a **custom trained model (`model.tflite`)** created using **Google Teachable Machine**.
- Trained with over **150+ images per class** listed in `labels.txt`.
- Detects real world objects like bottles, doors, chairs, etc.
- Outputs high confidence detections via audio (`pyttsx3`) and overlayed text.


### 📝 Text Detection (OCR)
- **Script:** `ocr.py`
- Captures image, sends it to a **paid OCR API** for accurate and fast text extraction.
- Raw OCR text is summarized via **Gemini API** to ensure concise and meaningful audio output.
- Alternatives:
  - You **can** use `pytesseract`, but it may be slower and less accurate.
  - Custom OCR models were avoided due to **high computational costs**.



---

## 🧠 Technical Choices

| Task              | Solution Used                                | Reason                                                  |
|-------------------|-----------------------------------------------|----------------------------------------------------------|
| Object Detection  | Teachable Machine + TFLite                    | Lightweight, real time performance on Raspberry Pi       |
| OCR               | Paid API                                      | Faster + more accurate than `pytesseract`               |
| Summarization     | Gemini 2.0 API (Google Generative AI)         | Makes OCR output concise for blind users                 |
| Voice Output      | `pyttsx3`                                     | Offline TTS for real time feedback                       |



---

## 🧪 File Structure

```bash
├── LICENSE
├── README.md
├── labels.txt              # Object class names
├── model.tflite            # Trained object detection model
├── obj_detection.py        # Real time object detection script
├── ocr.py                  # OCR capture + summarization + speech
├── requirements.txt        # Python dependencies
