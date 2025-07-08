from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
import torch
import torchvision.transforms as transforms
from PIL import Image
import io, sys, os, time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.satellite_cnn import SatelliteCNN

model = None
transform = None
class_names = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 
               'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 
               'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 
               'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 
               'storagetanks', 'tenniscourt']

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform
    print("Booting up the satellite terrain classification model 🛰️")
    
    model = SatelliteCNN(num_classes=21)
    model.load_state_dict(torch.load('notebooks/best_satellite_terrain_model.pth', map_location='cpu', weights_only=False))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Ready to go! 🛰️")

    yield

    print('Shutting down model')

app = FastAPI(title="Satellite Terrain Classifier", description="CNN for satellite imagery classification (96.43% accuracy)", 
              version="1.0.0", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🛰️ Satellite Terrain Classifier 🛰️</title>
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                max-width: 900px; margin: 0 auto; padding: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.95);
                color: #333;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .upload-area { 
                border: 3px dashed #667eea; 
                padding: 40px; 
                text-align: center; 
                margin: 30px 0;
                border-radius: 10px;
                background: #f8f9ff;
            }
            .stats {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 20px;
                margin: 30px 0;
            }
            .stat-box {
                background: #667eea;
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-number { font-size: 2em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s;
            }
            button:hover { background: #5a6fd8; transform: translateY(-2px); }
            .tech-stack {
                background: #f0f2f5;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛰️ Satellite Terrain Classifier 🛰️</h1>
            <p>CNN for satellite imagery classification (96.43% accuracy). Upload satellite imagery to classify terrain types using our ResNet50-based CNN.</p>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number">96.43%</div>
                    <div class="stat-label">Validation Accuracy</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">21</div>
                    <div class="stat-label">Terrain Classes</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">24.5M</div>
                    <div class="stat-label">Parameters</div>
                </div>
            </div>
            
            <div class="upload-area">
                <h3>🔍 Classify Satellite Image 🔍</h3>
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*" required style="margin: 10px;">
                    <br><br>
                    <button type="submit">🛰️ Analyze Terrain 🛰️</button>
                </form>
            </div>
            
            <div class="tech-stack">
                <h3>🔧 Technical Architecture 🔧</h3>
                <p><strong>Model:</strong> ResNet50 + Transfer Learning | <strong>Framework:</strong> PyTorch | <strong>API:</strong> FastAPI | <strong>Deployment:</strong> AWS Cloud</p>
                <p><strong>Training:</strong> UC Merced Land Use Dataset | <strong>Validation:</strong> Stratified 5-fold | <strong>Optimization:</strong> Adam + Learning Rate Scheduling</p>
            </div>
            
            <h3>📊 Supported Terrain Classifications 📊</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; font-size: 0.9em;">
                <div>• Agricultural</div><div>• Airplane</div><div>• Baseball Diamond</div>
                <div>• Beach</div><div>• Buildings</div><div>• Chaparral</div>
                <div>• Dense Residential</div><div>• Forest</div><div>• Freeway</div>
                <div>• Golf Course</div><div>• Harbor</div><div>• Intersection</div>
                <div>• Medium Residential</div><div>• Mobile Home Park</div><div>• Overpass</div>
                <div>• Parking Lot</div><div>• River</div><div>• Runway</div>
                <div>• Sparse Residential</div><div>• Storage Tanks</div><div>• Tennis Court</div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint for AWS load balancer"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_supported": len(class_names),
        "accuracy": "96.43%",
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict_terrain(file: UploadFile = File(...)):
    """Main prediction endpoint"""
    start_time = time.time()
    
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")
        
        # Process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Transform and predict
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Build comprehensive response
        results = {
            "status": "success",
            "prediction": {
                "class": class_names[predicted.item()],
                "confidence": round(float(confidence.item()), 4)
            },
            "top_3_predictions": [
                {
                    "class": class_names[idx.item()],
                    "probability": round(float(prob.item()), 4),
                    "confidence_level": "high" if prob > 0.8 else "medium" if prob > 0.5 else "low"
                }
                for idx, prob in zip(top3_indices, top3_prob)
            ],
            "metadata": {
                "filename": file.filename,
                "image_size": image.size,
                "inference_time_ms": round(inference_time, 2),
                "model_version": "ResNet50-satellite-v1.0",
                "accuracy": "96.43%"
            }
        }
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Detailed model information for technical documentation"""
    return {
        "model_name": "Satellite Terrain Classifier",
        "architecture": "ResNet50 + Custom Classification Head",
        "transfer_learning": "ImageNet pre-trained backbone",
        "performance": {
            "validation_accuracy": "96.43%",
            "training_accuracy": "90.65%",
            "dataset": "UC Merced Land Use (2,100 images)"
        },
        "technical_specs": {
            "input_size": "3x256x256 RGB",
            "output_classes": len(class_names),
            "total_parameters": "~24.5M",
            "trainable_parameters": "~1.5M",
            "framework": "PyTorch 2.0+"
        },
        "deployment": {
            "api_framework": "FastAPI",
            "inference_time": "~50-100ms",
            "cloud_platform": "AWS",
            "container": "Docker"
        },
        "classes": class_names
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)