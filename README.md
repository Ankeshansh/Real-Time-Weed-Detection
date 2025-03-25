# Real-Time-Weed-Detection

This project focuses on real-time weed detection using deep learning models, including **Faster R-CNN, RetinaNet, and SSD**. The model processes live camera feeds to classify whether a plant is a weed or not, enabling smart agricultural solutions.  

🔗 **Live Demo**: [Weed Detection App](https://real-time-weed-detection-ahth9azxfnewlmqphufecz.streamlit.app/)

## 🚀 Features  
- **Deep Learning Models**: Utilizes Faster R-CNN, RetinaNet, and SSD for accurate weed detection.  
- **Live Stream Detection**: Implements a real-time detection system using Google Colab's camera.  
- **Web Deployment**: Deployed via **Streamlit** for easy accessibility.  
- **Performance Metrics**: Compared models based on the **mean Dice coefficient (0.65)**.  

## 📁 Dataset  
- Images of different types of weeds and crops.  
- Augmented dataset to improve model robustness.  

## 🏗️ Model Pipeline  
1. **Data Preprocessing**: Normalization, augmentation, and resizing.  
2. **Training**: Model trained on a labeled dataset with PyTorch.  
3. **Evaluation**: Compared the performance of all three models.  
4. **Deployment**: Integrated into a web-based application using Streamlit.

## 📊 Model Performance  

| Model       | Mean Dice Coefficient |
|------------|----------------------|
| Faster R-CNN | 0.67 |
| RetinaNet   | 0.65 |
| SSD         | 0.63 |

## 📌 Installation  
Clone the repository and install dependencies:  
```bash
  git clone https://github.com/your-repo/real-time-weed-detection.git
  cd real-time-weed-detection
  pip install -r requirements.txt


