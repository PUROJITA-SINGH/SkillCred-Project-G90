# Skillcred-Project-G90
PriceAI & Social Donation Website 🌟
🚀 Welcome to the Main Branch!This repository combines two powerful projects: PriceAI Backend, an e-commerce price prediction system, and Social Donation Website, a modern one-page donation platform. Both projects showcase advanced technologies and seamless user experiences, catering to different use cases in e-commerce and social impact.

📋 Overview
PriceAI Backend
PriceAI Backend is a comprehensive Python pipeline for e-commerce price prediction, leveraging machine learning, web scraping, and real-time market analysis. It integrates with a frontend (not included) to deliver accurate pricing intelligence for products across platforms like Amazon, eBay, and Walmart.
Social Donation Website
The Social Donation Website is a single-page, responsive web application designed for seamless donations. It uses Tailwind CSS for styling, Stripe for secure payments, and supports integration with a headless CMS and a Large Language Model (LLM) for personalized donor communication.

✨ Features
PriceAI Backend

Advanced ML Models: Ensemble of Random Forest, Gradient Boosting, Ridge, and Elastic Net for accurate price predictions.
Web Scraping: Asynchronous scraping of Amazon, eBay, and Walmart for competitor pricing and product data.
Feature Engineering: Text analysis (sentiment, keywords), categorical encoding (category, brand, condition), and scraped data features.
Image Analysis: Optional computer vision for product images using ResNet50 (requires PyTorch).
REST API: Flask-based API with endpoints for predictions, batch processing, training, stats, and market trends.
Database Management: SQLite for storing products, predictions, and scraped data.
Batch Processing: Supports CSV uploads for bulk price predictions.

Social Donation Website

Single-Page Design: Clean, intuitive UI built with HTML and Tailwind CSS.
Secure Payments: Stripe integration for reliable donation processing.
Headless CMS: Designed for easy content updates via a CMS (e.g., Sanity, Contentful).
Personalized Communication: Backend (not included) uses an LLM to generate thank-you emails and impact summaries.
Interactive Impact Meter: Real-time visual feedback on donation impact.
Responsive UI: Optimized for desktop and mobile devices.


💻 Technologies Used
PriceAI Backend

Core: Python, Flask, SQLite
Scraping: requests, aiohttp, BeautifulSoup, Selenium
Machine Learning: scikit-learn, numpy, pandas
Text Processing: NLTK, TextBlob, spacy
Image Analysis: PyTorch, torchvision (optional)
Dependencies: selenium, flask-cors, joblib, nltk, etc.

Social Donation Website

Frontend: HTML, JavaScript, Tailwind CSS
Payments: Stripe JS v3
Content Management: Headless CMS (e.g., Sanity, Contentful, Strapi)
Backend (not included): LLM (e.g., GPT-3/4, PaLM), email service (e.g., SendGrid)



📄 Code Structure
PriceAI Backend

main.py: Core script with all classes (WebScraper, FeatureExtractor, PricePredictionModel, DatabaseManager, PriceAIAPI, ImageAnalyzer, DataValidator) and CLI interface.
Database: SQLite file (priceai.db) for storing data.
Model: Saved as priceai_model.pkl using joblib.

Social Donation Website

index.html: Single file containing HTML, Tailwind CSS, Stripe.js, and JavaScript for navigation, donation forms, and impact meter.
Structure:
<head>: Includes Tailwind CSS and Stripe.js CDNs, custom styles.
<body>: Two sections (homepage, donate) for single-page navigation.
<script>: Handles client-side logic and Stripe integration.




😊 Notes

PriceAI Backend: Scalable with asynchronous scraping and robust error handling. Requires significant computational resources for scraping and ML training.
Social Donation Website: Lightweight frontend with no build steps. Backend setup is required for full functionality.
Contributions: Feel free to fork, submit PRs, or report issues!
License: MIT License (update as needed).

😎 Thank You for Exploring PriceAI & Social Donation Website!
