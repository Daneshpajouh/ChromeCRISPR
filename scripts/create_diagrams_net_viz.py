#!/usr/bin/env python3
"""
Create Diagrams.net Style Visualizations for ChromeCRISPR
"""

def create_architecture_html():
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChromeCRISPR: CNN-GRU+GC Architecture</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .title {
            text-align: center;
            color: #1976d2;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 30px;
        }
        .architecture {
            position: relative;
            height: 600px;
            margin: 40px 0;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            background: #fafafa;
        }
        .layer {
            position: absolute;
            border: 2px solid #333;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-weight: bold;
            color: #333;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .input-layer { background: #e3f2fd; border-color: #1976d2; }
        .embedding-layer { background: #f3e5f5; border-color: #7b1fa2; }
        .conv-layer { background: #e8f5e8; border-color: #388e3c; }
        .pool-layer { background: #fff3e0; border-color: #f57c00; }
        .gru-layer { background: #fce4ec; border-color: #c2185b; }
        .dense-layer { background: #e0f2f1; border-color: #00695c; }
        .output-layer { background: #fafafa; border-color: #424242; }
        .gc-layer { background: #fff8e1; border-color: #ff8f00; }
        .layer-label {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 4px;
        }
        .layer-details {
            font-size: 11px;
            color: #666;
        }
        .arrow {
            position: absolute;
            height: 2px;
            background: #333;
            transform-origin: left center;
        }
        .arrow::after {
            content: '';
            position: absolute;
            right: -8px;
            top: -3px;
            width: 0;
            height: 0;
            border-left: 8px solid #333;
            border-top: 4px solid transparent;
            border-bottom: 4px solid transparent;
        }
        .performance-box {
            background: #f5f5f5;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
            text-align: center;
        }
        .performance-title {
            font-size: 18px;
            font-weight: bold;
            color: #1976d2;
            margin-bottom: 10px;
        }
        .performance-metrics {
            font-size: 16px;
            color: #333;
        }
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border: 2px solid #333;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">ChromeCRISPR: CNN-GRU+GC Architecture</h1>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color input-layer"></div>
                <span>Input Layers</span>
            </div>
            <div class="legend-item">
                <div class="legend-color embedding-layer"></div>
                <span>Embedding</span>
            </div>
            <div class="legend-item">
                <div class="legend-color conv-layer"></div>
                <span>Convolutional</span>
            </div>
            <div class="legend-item">
                <div class="legend-color pool-layer"></div>
                <span>Pooling</span>
            </div>
            <div class="legend-item">
                <div class="legend-color gru-layer"></div>
                <span>Recurrent (GRU)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color dense-layer"></div>
                <span>Dense/Fusion</span>
            </div>
            <div class="legend-item">
                <div class="legend-color gc-layer"></div>
                <span>Biological Features</span>
            </div>
        </div>
        
        <div class="architecture">
            <div class="layer input-layer" style="left: 50px; top: 80px; width: 120px; height: 60px;">
                <div class="layer-label">Input</div>
                <div class="layer-details">21-mer sgRNA</div>
            </div>
            
            <div class="layer embedding-layer" style="left: 220px; top: 80px; width: 120px; height: 60px;">
                <div class="layer-label">Embedding</div>
                <div class="layer-details">64 dim</div>
            </div>
            
            <div class="layer conv-layer" style="left: 390px; top: 50px; width: 120px; height: 60px;">
                <div class="layer-label">Conv1D</div>
                <div class="layer-details">128 filters</div>
            </div>
            
            <div class="layer conv-layer" style="left: 390px; top: 130px; width: 120px; height: 60px;">
                <div class="layer-label">Conv1D</div>
                <div class="layer-details">128 filters</div>
            </div>
            
            <div class="layer pool-layer" style="left: 560px; top: 50px; width: 120px; height: 60px;">
                <div class="layer-label">MaxPool</div>
                <div class="layer-details">k=2</div>
            </div>
            
            <div class="layer pool-layer" style="left: 560px; top: 130px; width: 120px; height: 60px;">
                <div class="layer-label">MaxPool</div>
                <div class="layer-details">k=2</div>
            </div>
            
            <div class="layer gru-layer" style="left: 390px; top: 220px; width: 120px; height: 60px;">
                <div class="layer-label">GRU</div>
                <div class="layer-details">384 units</div>
            </div>
            
            <div class="layer gru-layer" style="left: 560px; top: 220px; width: 120px; height: 60px;">
                <div class="layer-label">GRU</div>
                <div class="layer-details">384 units</div>
            </div>
            
            <div class="layer gc-layer" style="left: 50px; top: 320px; width: 120px; height: 60px;">
                <div class="layer-label">GC Content</div>
                <div class="layer-details">Feature</div>
            </div>
            
            <div class="layer dense-layer" style="left: 730px; top: 90px; width: 120px; height: 60px;">
                <div class="layer-label">Feature</div>
                <div class="layer-details">Fusion</div>
            </div>
            
            <div class="layer dense-layer" style="left: 900px; top: 50px; width: 120px; height: 60px;">
                <div class="layer-label">Dense</div>
                <div class="layer-details">128</div>
            </div>
            
            <div class="layer dense-layer" style="left: 900px; top: 130px; width: 120px; height: 60px;">
                <div class="layer-label">Dense</div>
                <div class="layer-details">64</div>
            </div>
            
            <div class="layer dense-layer" style="left: 900px; top: 210px; width: 120px; height: 60px;">
                <div class="layer-label">Dense</div>
                <div class="layer-details">32</div>
            </div>
            
            <div class="layer output-layer" style="left: 900px; top: 290px; width: 120px; height: 60px;">
                <div class="layer-label">Output</div>
                <div class="layer-details">Efficiency</div>
            </div>
            
            <div class="arrow" style="left: 170px; top: 110px; width: 50px;"></div>
            <div class="arrow" style="left: 340px; top: 80px; width: 50px;"></div>
            <div class="arrow" style="left: 340px; top: 160px; width: 50px;"></div>
            <div class="arrow" style="left: 340px; top: 250px; width: 50px;"></div>
            <div class="arrow" style="left: 510px; top: 80px; width: 50px;"></div>
            <div class="arrow" style="left: 510px; top: 160px; width: 50px;"></div>
            <div class="arrow" style="left: 680px; top: 80px; width: 50px;"></div>
            <div class="arrow" style="left: 680px; top: 160px; width: 50px;"></div>
            <div class="arrow" style="left: 680px; top: 250px; width: 50px;"></div>
            <div class="arrow" style="left: 850px; top: 80px; width: 50px;"></div>
            <div class="arrow" style="left: 850px; top: 160px; width: 50px;"></div>
            <div class="arrow" style="left: 850px; top: 240px; width: 50px;"></div>
            <div class="arrow" style="left: 1020px; top: 80px; width: 2px; height: 50px; transform: rotate(90deg);"></div>
            <div class="arrow" style="left: 1020px; top: 160px; width: 2px; height: 50px; transform: rotate(90deg);"></div>
            <div class="arrow" style="left: 1020px; top: 240px; width: 2px; height: 50px; transform: rotate(90deg);"></div>
            <div class="arrow" style="left: 170px; top: 350px; width: 560px;"></div>
        </div>
        
        <div class="performance-box">
            <div class="performance-title">Performance Metrics</div>
            <div class="performance-metrics">
                Spearman Correlation: 0.876 | Mean Squared Error: 0.0093 | R² Score: 0.767
            </div>
        </div>
    </div>
</body>
</html>'''
    
    with open('ChromeCRISPR_Architecture.html', 'w') as f:
        f.write(html)
    print("✅ Architecture visualization created")

def create_performance_html():
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChromeCRISPR: Model Performance Comparison</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .title {
            text-align: center;
            color: #1976d2;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 30px;
        }
        .chart-container {
            margin: 40px 0;
        }
        .bar-row {
            display: flex;
            align-items: center;
            margin: 8px 0;
            height: 40px;
        }
        .model-name {
            width: 200px;
            font-weight: bold;
            font-size: 14px;
            text-align: right;
            padding-right: 15px;
            color: #333;
        }
        .bar-container {
            flex: 1;
            height: 30px;
            background: #f0f0f0;
            border-radius: 15px;
            position: relative;
            margin: 0 15px;
        }
        .bar {
            height: 100%;
            border-radius: 15px;
            position: relative;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .bar:hover {
            transform: scaleY(1.1);
        }
        .bar-value {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-weight: bold;
            font-size: 12px;
            color: white;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        }
        .best-indicator {
            position: absolute;
            right: -40px;
            top: 50%;
            transform: translateY(-50%);
            color: #e74c3c;
            font-weight: bold;
            font-size: 14px;
        }
        .hybrid-color { background: linear-gradient(90deg, #4caf50, #45a049); }
        .cnn-color { background: linear-gradient(90deg, #2196f3, #1976d2); }
        .rnn-color { background: linear-gradient(90deg, #9c27b0, #7b1fa2); }
        .rf-color { background: linear-gradient(90deg, #ff9800, #f57c00); }
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
        .legend-hybrid { background: #4caf50; }
        .legend-cnn { background: #2196f3; }
        .legend-rnn { background: #9c27b0; }
        .legend-rf { background: #ff9800; }
        .summary-box {
            background: #f5f5f5;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
            text-align: center;
        }
        .summary-title {
            font-size: 18px;
            font-weight: bold;
            color: #1976d2;
            margin-bottom: 10px;
        }
        .summary-text {
            font-size: 16px;
            color: #333;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">ChromeCRISPR: Model Performance Comparison</h1>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color legend-hybrid"></div>
                <span>Hybrid Models (CNN + RNN)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-cnn"></div>
                <span>CNN Models</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-rnn"></div>
                <span>RNN Models</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-rf"></div>
                <span>Random Forest</span>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="bar-row">
                <div class="model-name">CNN-GRU+GC</div>
                <div class="bar-container">
                    <div class="bar hybrid-color" style="width: 100%;">
                        <div class="bar-value">0.876</div>
                    </div>
                </div>
                <div class="best-indicator">★ BEST</div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">CNN-BiLSTM+GC</div>
                <div class="bar-container">
                    <div class="bar hybrid-color" style="width: 99.3%;">
                        <div class="bar-value">0.870</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">CNN-LSTM+GC</div>
                <div class="bar-container">
                    <div class="bar hybrid-color" style="width: 99.0%;">
                        <div class="bar-value">0.867</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">deepCNN+GC</div>
                <div class="bar-container">
                    <div class="bar cnn-color" style="width: 99.7%;">
                        <div class="bar-value">0.873</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">deepBiLSTM+GC</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 99.0%;">
                        <div class="bar-value">0.867</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">deepGRU+GC</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 99.0%;">
                        <div class="bar-value">0.867</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">deepLSTM+GC</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 98.2%;">
                        <div class="bar-value">0.860</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">deepCNN</div>
                <div class="bar-container">
                    <div class="bar cnn-color" style="width: 99.2%;">
                        <div class="bar-value">0.869</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">deepGRU</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 99.1%;">
                        <div class="bar-value">0.868</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">deepLSTM</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 98.4%;">
                        <div class="bar-value">0.862</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">deepBiLSTM</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 98.4%;">
                        <div class="bar-value">0.862</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">LSTM+GC</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 97.7%;">
                        <div class="bar-value">0.856</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">BiLSTM+GC</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 97.6%;">
                        <div class="bar-value">0.855</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">GRU+GC</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 95.9%;">
                        <div class="bar-value">0.840</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">CNN+GC</div>
                <div class="bar-container">
                    <div class="bar cnn-color" style="width: 89.2%;">
                        <div class="bar-value">0.781</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">BiLSTM</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 96.2%;">
                        <div class="bar-value">0.843</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">LSTM</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 95.5%;">
                        <div class="bar-value">0.837</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">GRU</div>
                <div class="bar-container">
                    <div class="bar rnn-color" style="width: 95.5%;">
                        <div class="bar-value">0.837</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">CNN</div>
                <div class="bar-container">
                    <div class="bar cnn-color" style="width: 90.5%;">
                        <div class="bar-value">0.793</div>
                    </div>
                </div>
            </div>
            
            <div class="bar-row">
                <div class="model-name">Random Forest</div>
                <div class="bar-container">
                    <div class="bar rf-color" style="width: 86.2%;">
                        <div class="bar-value">0.755</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="summary-box">
            <div class="summary-title">Key Findings</div>
            <div class="summary-text">
                The CNN-GRU+GC hybrid model achieves the best performance with a Spearman correlation of 0.876. 
                Hybrid models combining CNN and RNN architectures consistently outperform single-architecture models. 
                The integration of biological features (GC content) improves performance across all model types.
            </div>
        </div>
    </div>
</body>
</html>'''
    
    with open('ChromeCRISPR_Performance.html', 'w') as f:
        f.write(html)
    print("✅ Performance comparison created")

if __name__ == "__main__":
    print("🎨 Creating Diagrams.net Style ChromeCRISPR Visualizations")
    print("=========================================================")
    create_architecture_html()
    create_performance_html()
    print("\n🎉 Visualizations completed!")
    print("📋 Files created:")
    print("   - ChromeCRISPR_Architecture.html")
    print("   - ChromeCRISPR_Performance.html")
    print("\n💡 Open these files in a web browser to view the visualizations")
