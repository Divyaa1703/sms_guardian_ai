🚀📱 SMS Guardian AI
Intelligent Multilingual Spam Detection System using Machine Learning & NLP










🌟 Project Overview

SMS Guardian AI is an advanced Machine Learning and Natural Language Processing (NLP) based spam detection system that intelligently classifies SMS messages as Spam or Not Spam.

The system is specially designed to protect users from:

🚨 Fraud messages
🔗 Phishing links
💰 Fake offers
📢 Promotional spam

Unlike traditional spam filters, SMS Guardian AI uses intelligent text analysis to understand message patterns and predict spam with high accuracy.

✨ Key Features
📩 Real-Time Spam Detection

Users can instantly check whether a message is spam or safe.

🌐 Multi-Language Support

Supports:

🇬🇧 English
🇮🇳 Hindi
🇮🇳 Marathi

👉 Makes the project practical for real-world Indian users.

📂 Bulk File Classification

Users can upload:

CSV files
TXT files

for large-scale spam analysis.

📥 Export Results

Download classified results in:

CSV
Excel format

Includes:

Message
Prediction
Confidence Score
👨‍💻 Advanced Admin Panel
✅ User Management

Admin can:

View registered users
Block/Delete users
Monitor user activity
✅ Classification Monitoring

Admin can track:

Messages
Predictions
Languages
URLs
Confidence Scores
✅ Feedback Reports

Admin can:

View user feedback
Mark reports as reviewed
📊 Analytics Dashboard

Interactive dashboard showing:

Total Messages
Spam Count
Non-Spam Count
Spam Statistics
Graphs & Charts
💬 Feedback System

Users can report wrong predictions to improve future model performance.

🛠️ Technologies Used
👨‍💻 Programming Language
Python
🤖 Machine Learning & NLP
Scikit-learn
NLTK
TF-IDF Vectorizer
Multinomial Naïve Bayes
🌐 Web Technologies
Streamlit
HTML
CSS
🗄️ Database
SQLite
⚙️ System Workflow
User Input
   ↓
Text Preprocessing
   ↓
Language Detection
   ↓
TF-IDF Vectorization
   ↓
Naïve Bayes Model
   ↓
Spam Prediction
   ↓
Database Storage
   ↓
Result & Analytics
🧠 Machine Learning Pipeline
📌 Data Preprocessing

The system cleans SMS text using:

Lowercase conversion
Tokenization
Stopword removal
Stemming
📌 Feature Extraction

TF-IDF converts text into numerical vectors for model training.

📌 Model Training

The system uses the Multinomial Naïve Bayes algorithm because:

Fast performance
High accuracy
Best for text classification
🗄️ Database Design
👤 Users Table
Field	Description
id	User ID
username	Username
email	Email
role	User/Admin
created_at	Registration Time
📩 Classification Table
Field	Description
message	SMS content
prediction	Spam/Not Spam
confidence	Prediction confidence
language	Detected language
timestamp	Classification time
💬 Feedback Table
Field	Description
comment	User feedback
status	Pending/Reviewed
timestamp	Feedback time
📈 Model Performance
Metric	Result
Accuracy	95–98%
Precision	High
Recall	High
Prediction Speed	Fast
📷 Main Modules

✅ Registration & Login
✅ Spam Classification
✅ Bulk File Upload
✅ Classification History
✅ Admin Dashboard
✅ Analytics
✅ Feedback Reports
✅ Result Export

🚀 Future Enhancements
🔥 Deep Learning Models (LSTM/BERT)
📱 Mobile Application
🌍 More Language Support
☁️ Cloud Deployment
🔄 Auto Model Retraining
▶️ Installation & Setup
📌 Clone Repository
git clone <repository-link>
📌 Install Dependencies
pip install -r requirements.txt
📌 Run Application
streamlit run app.py
🎯 Project Highlights

✔ Machine Learning Based
✔ NLP Integration
✔ Multi-language Support
✔ Admin Dashboard
✔ Bulk Classification
✔ Real-Time Prediction
✔ Database Integration
✔ Analytics & Visualization

👩‍💻 Developed By
Divya Gore

🎓 MCA Student
💡 Passionate about Data Analytics & Machine Learning

⭐ Conclusion

SMS Guardian AI is a smart, scalable, and user-friendly spam detection system that combines Machine Learning, NLP, and Web Technologies to provide secure and intelligent SMS classification.

The project successfully solves a real-world problem by helping users identify spam messages quickly and efficiently while improving digital communication safety.
