# 🌱 Plant Disease Detection & Plant Identification Chatbot

A powerful AI-based chatbot that helps farmers and gardeners identify **plant species** and **plant diseases** from images. Built with PyTorch, ResNet18, and a friendly interactive interface.

---

## ✨ Features

- **Plant Species Detection** - Identify 30+ common crops and plants
- **Plant Disease Detection** - Detect various diseases in Cotton, Rice, Wheat, Sugarcane, Maize, etc.
- **Combined Detection** - Identify both plant type and disease in one go
- **Interactive Chatbot** - Ask questions in natural English
- **Fun Facts & Tips** - Learn interesting facts and get practical care advice
- **Plant Quiz** - Test your knowledge with 20+ quiz questions
- **History Tracking** - View your recent queries
- **Color-coded Confidence** - Green (High), Yellow (Medium), Red (Low)

---

## 🖼️ Supported Plants & Crops

- Tomato, Potato, Banana, Coconut, Rice, Wheat, Maize, Cotton, Sugarcane, and many more (30+ classes)

**Diseases covered**: American Bollworm, Rice Blast, Wheat Rust, Red Rot, Pink Bollworm, Leaf Curl, and many others.

---

## 🚀 How to Use

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow fuzzywuzzy python-Levenshtein
