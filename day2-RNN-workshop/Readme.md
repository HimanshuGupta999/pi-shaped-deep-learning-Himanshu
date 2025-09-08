# Recurrent Neural Networks (RNN) 

## Project Overview

This repository contains a Jupyter Notebook (`RNN.ipynb`) that build and train an LSTM-based RNN to predict Tesla’s next-day opening stock price using historical data.

### Methodology

The project follows these key steps:
1.  **Data Loading**: The Tesla stock price dataset is loaded from CSV for analysis.
2.  **Preprocessing**: The `Open` prices are scaled using `MinMaxScaler`, and the data is framed into supervised sequences (e.g., 60 past days → next day’s price). The dataset is then split into training and testing sets.
3.  **Model Building**: A sequence-based RNN model is constructed using TensorFlow/Keras, specifically with stacked LSTM layers and dropout regularization.
4.  **Training**: The model is trained on historical stock sequences using the Adam optimizer and mean squared error (MSE) as the loss function.
5.  **Evaluation**: The model’s predictive performance is assessed on the test set using regression metrics such as RMSE and MAE, and predictions are visualized against actual stock prices.

### How to Run This Project

1.  **Prerequisites**: Make sure you have Python installed, along with the following libraries:
    *   numpy
    *   pandas
    *   scikit-learn
    *   tensorflow
    *   matplotlib
    *   jupyter

    You can install them using pip:
    ```bash
    pip install -r requirements.txt    
    ```

2.  **Execution**: Open and run the `RNN.ipynb` file in a Jupyter environment. The notebook is self-contained and includes explanations for each step.

## Core Concepts Questions

### 1. What is the benefit of using RNNs (or LSTMs) over traditional feedforward networks for time-series data?  
- Feedforward networks treat inputs as independent, ignoring temporal order.  
- RNNs/LSTMs are designed to handle sequential dependencies, meaning they can learn patterns that unfold over time (e.g., stock prices today depend on past trends).  
- LSTMs, in particular, solve the *vanishing gradient problem*, enabling learning of long-term dependencies.  

### 2. Why is sequence framing (input windows) important in time series forecasting?  
- Raw time series is just a continuous stream of values.  
- **Framing into input windows** (e.g., past 60 days → predict next day) transforms it into a supervised learning problem.  
- Without framing, the model wouldn’t know how much history to consider for predicting the future.  

### 3. How does feature scaling impact training of RNN/LSTM models?  
- Stock prices vary widely in scale (e.g., Tesla could range from $50 to $1200).  
- RNNs/LSTMs use gradient-based optimization; large unscaled values can lead to unstable training.  
- **MinMax scaling (0–1)** helps models converge faster and avoid exploding gradients.  

### 4. Compare SimpleRNN and LSTM in terms of handling long-term dependencies.  
- **SimpleRNN**: Captures short-term dependencies but struggles with long sequences due to vanishing/exploding gradients.  
- **LSTM**: Uses gates (input, forget, output) to selectively retain or discard information, making it far better at modeling **long-term dependencies**.  

### 5. What regression metrics (e.g., MAE, RMSE) are appropriate for stock price prediction, and why?  
- **MAE (Mean Absolute Error)**: Measures average absolute difference → easy to interpret in dollar terms.  
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more strongly, which is useful when big prediction errors are costly.  
- Both are common in financial forecasting; RMSE is often preferred when large deviations are especially risky.  

### 6. How can you assess if your model is overfitting?  
- Compare `training vs validation loss` during training.  
- If training loss keeps decreasing while validation loss increases, the model is memorizing instead of generalizing.  
- Visual inspection of predictions vs actual values can also reveal overfitting (too “perfect” on training data but poor on unseen data).  

### 7. How might you extend the model to improve performance (e.g., more features, deeper network)?  
- Include additional features: `High, Low, Close, Volume, Technical Indicators`.  
- Use a deeper network (stacked LSTMs or GRUs).  
- Try regularization (Dropout, L2 penalties) or early stopping.  
- Hyperparameter tuning (learning rate, window size, units per layer).  
- Use hybrid architectures (CNN + LSTM, Transformer-based models).  

### 8. Why is it important to shuffle (or not shuffle) sequential data during training?  
- In **time series forecasting**, order matters. Shuffling would break temporal dependencies.  
- Typically, we **don’t shuffle training data**. Instead, we keep it sequential.  
- However, within **mini-batches**, shuffling may sometimes be applied to avoid bias, but order across sequences must be preserved.  

### 9. How can you visualize model prediction vs actual values to interpret performance?  
- Plot actual stock prices vs predicted prices over time.  
- This shows how closely predictions follow the trend (alignment and lag).  
- Residual plots (errors vs time) can also highlight systematic biases.  

### 10. What real-world challenges arise when using RNNs for stock price prediction?  
- **Market noise**: Stock prices are influenced by external events (news, regulations) not captured in historical data.  
- **Overfitting**: Easy to fit past patterns that don’t generalize.  
- **Non-stationarity**: Market behavior changes over time.  
- **Data availability**: Missing values, stock splits, or irregular trading days complicate modeling.  
- **Interpretability**: Hard to explain model decisions in a high-stakes financial domain.  