# Basic Neural Network

This project implements a simple neural network from scratch using NumPy to classify flowers based on two inputs (theirs width and length).  

## 📌 Features  
- A basic feedforward neural network with one hidden layer.  
- Uses sigmoid activation and backpropagation for training.  
- Predicts the color of a flower (Red or Blue) based on **2 input values**.
- The datas of this project are totally fictive

## 🛠️ Requirements  
- Python 3.x  
- NumPy  

## 🚀 Usage  
Clone the repository and run the script
It will train the neural network and make a prediction based on a test input. 

## 🧠 How It Works  
1. **Data Preprocessing**:  
   - Normalizes the input data.  
   - Splits it into training data and a separate test input.  

2. **Neural Network Structure**:  
   - Input layer: 2 neurons  
   - Hidden layer: 3 neurons  
   - Output layer: 1 neuron (predicting 0 or 1)  

3. **Training**:  
   - Uses backpropagation with gradient descent for weight updates.  
   - Runs for 1,000,000 iterations to minimize error.  

4. **Prediction**:  
   - After training, the network predicts the color of a flower based on new input data.
   - The flower is blue if y=0 or red if y=1

## 📚 Reference  
Inspired by this YouTube tutorial: [Link](https://www.youtube.com/watch?v=bzC5cdxZcOM&t=446s).  

## ⏭️ **Next Step**
How can we classify the flowers if we have a third color (green for exemple)

## 👤 Author  
Camille ANSEL

