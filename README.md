# Multi-Attention Image Caption API - Deployment Guide

## Overview

This deployment creates a FastAPI + Modal stack that:
- Uploads shared dataset artifacts (captions, tokenizer, precomputed features) to a Modal volume
- Remotely trains every attention variant through `train_multi_attention.py`
- Saves the resulting `.keras` checkpoints in the same volume for reuse
- Exposes REST APIs (with CORS enabled) so frontend apps such as Lovable can list available models and request captions

## Supported Attention Types

1. **luong** - Luong attention (dot-product)
2. **bahdanau** - Bahdanau additive attention
3. **scaled_dot** - Scaled dot-product attention
4. **multihead** - Multi-head attention
5. **concatenate** - Concatenation-based attention

## Deployment Steps

### 1. Prerequisites

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal token new
```

### 2. Upload Dataset Files

Upload your preprocessed artifacts to Modal's persistent volume:

```bash
modal run deploy.py::upload_dataset
```

This uploads:
- `results/captions.csv` - cleaned caption dataset (pipe-delimited)
- `results/img_features.pkl` - VGG16 image features
- `results/tokenizer.pkl` - Pre-fitted tokenizer

### 3. Train Models

Launch remote training (defaults to all attention types, batch size 256, 20 epochs):

```bash
modal run deploy.py
```

Customize via CLI flags, e.g.:

```bash
# Only train luong and multihead for 10 epochs
modal run deploy.py --attention luong,multihead --epochs 10
```

Trained models are written to `/data/models` in the Modal volume.

Example run output (all five attention variants):

```
Restoring model weights from the end of the best epoch: 19.
Saved model to /data/models/caption_concatenate.keras

Trained model artifacts:
 - luong: /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_luong.keras
 - bahdanau: /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_bahdanau.keras
 - scaled_dot: /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_scaled_dot.keras
 - multihead: /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_multihead.keras
 - concatenate: /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_concatenate.keras

Model comparison
Attention      Best Ep Best Val    Final Val   Model
----------------------------------------------------
luong          17      4.0518      4.0660      /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_luong.keras
bahdanau       20      4.2271      4.2271      /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_bahdanau.keras
scaled_dot     20      4.0399      4.0399      /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_scaled_dot.keras
multihead      20      4.0717      4.0717      /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_multihead.keras
concatenate    19      3.9074      3.9075      /__modal/volumes/vo-x01NtEs7qtFCdjcVhRovja/models/caption_concatenate.keras

Training run complete. Summary:
- luong: best_val_loss=4.0518
- bahdanau: best_val_loss=4.2271
- scaled_dot: best_val_loss=4.0399
- multihead: best_val_loss=4.0717
- concatenate: best_val_loss=3.9074
```

### 4. Deploy the API

Deploy the FastAPI service to Modal:

```bash
modal deploy deploy.py
```

After deployment, Modal will provide a URL like:
```
https://your-username--image-caption-attention-trainer-fastapi-app.modal.run
```

**Save this URL** - you'll use it in your Lovable app!

## API Endpoints

### Base URL
```
https://your-username--image-caption-api-fastapi-app.modal.run
```

### 1. List Available Models
```http
GET /models
```

Response:
```json
{
  "models": [
    {"attention": "luong", "path": "/data/models/caption_luong.keras"},
    {"attention": "bahdanau", "path": "/data/models/caption_bahdanau.keras"}
  ]
}
```

### 2. Generate Caption (Main Endpoint)
```http
POST /generate-caption
Content-Type: multipart/form-data

Parameters:
- file: image file (JPEG, PNG)
- attention_type: "luong" | "bahdanau" | "scaled_dot" | "multihead" | "concatenate"
```

Response:
```json
{
  "caption": "a dog sitting on a beach",
  "status": "success",
  "attention_type": "luong"
}
```

### 3. Root Info
```http
GET /
```

Response:
```json
{
  "status": "ok",
  "message": "Image captioning API running on Modal.",
  "models": [
    {"attention": "luong", "path": "/data/models/caption_luong.keras"},
    ...
  ],
  "endpoints": {...}
}
```

### 4. Train Model (Admin)
```http
POST /train-model
Content-Type: application/x-www-form-urlencoded

Parameters:
- attention_type: model type to train
```

Response:
```json
{
  "status": "training_started",
  "attention_type": "luong",
  "message": "Training luong model started in background",
  "call_id": "call_xyz123"
}
```

### 5. Training Status
```http
GET /training-status
```

Response:
```json
{
  "status": "success",
  "trained_models": 5,
  "total_possible": 5,
  "models": [...]
}
```

## Lovable App Integration

### Frontend Code Example (React)

```typescript
import { useState } from 'react';

const API_URL = 'https://your-username--image-caption-api-fastapi-app.modal.run';

const ATTENTION_TYPES = [
  { value: 'luong', label: 'Luong Attention' },
  { value: 'bahdanau', label: 'Bahdanau Attention' },
  { value: 'scaled_dot', label: 'Scaled Dot-Product' },
  { value: 'multihead', label: 'Multi-Head Attention' },
  { value: 'concatenate', label: 'Concatenation' }
];

export default function ImageCaptionGenerator() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [attentionType, setAttentionType] = useState('luong');
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);

  const handleGenerateCaption = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('attention_type', attentionType);

    try {
      const response = await fetch(`${API_URL}/generate-caption`, {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setCaption(data.caption);
      }
    } catch (error) {
      console.error('Error generating caption:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Image Caption Generator</h1>
      
      {/* Model Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">
          Attention Model
        </label>
        <select
          value={attentionType}
          onChange={(e) => setAttentionType(e.target.value)}
          className="w-full p-2 border rounded"
        >
          {ATTENTION_TYPES.map((type) => (
            <option key={type.value} value={type.value}>
              {type.label}
            </option>
          ))}
        </select>
      </div>

      {/* File Upload */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">
          Upload Image
        </label>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
          className="w-full p-2 border rounded"
        />
      </div>

      {/* Preview */}
      {selectedFile && (
        <div className="mb-4">
          <img
            src={URL.createObjectURL(selectedFile)}
            alt="Preview"
            className="max-w-full h-auto rounded"
          />
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={handleGenerateCaption}
        disabled={!selectedFile || loading}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-400"
      >
        {loading ? 'Generating...' : 'Generate Caption'}
      </button>

      {/* Caption Result */}
      {caption && (
        <div className="mt-6 p-4 bg-gray-100 rounded">
          <p className="text-sm text-gray-600">Generated Caption:</p>
          <p className="text-lg font-medium italic">"{caption}"</p>
          <p className="text-xs text-gray-500 mt-2">Model: {attentionType}</p>
        </div>
      )}
    </div>
  );
}
```

### Using cURL

```bash
# Generate caption with Luong attention
curl -X POST "https://your-username--image-caption-api-fastapi-app.modal.run/generate-caption" \
  -F "file=@/path/to/image.jpg" \
  -F "attention_type=luong"

# List available models
curl "https://your-username--image-caption-api-fastapi-app.modal.run/models"

# Health check
curl "https://your-username--image-caption-api-fastapi-app.modal.run/health"
```

### Using Python

```python
import requests

API_URL = "https://your-username--image-caption-api-fastapi-app.modal.run"

# Generate caption
with open("image.jpg", "rb") as f:
    files = {"file": f}
    data = {"attention_type": "multihead"}
    response = requests.post(f"{API_URL}/generate-caption", files=files, data=data)
    
result = response.json()
print(f"Caption: {result['caption']}")
print(f"Model: {result['model']}")
```

## Configuration

Training knobs live in `train_multi_attention.py` (see `TrainingConfig`). Override them via CLI flags when invoking `modal run deploy.py`, e.g. `--batch-size 128 --learning-rate 5e-4 --epochs 30`. The API always serves the latest checkpoints saved in `/data/models`.

## Monitoring

### Check Logs
```bash
# View app logs
modal app logs image-caption-api

# View specific function logs
modal function logs image-caption-api.train_single_model
```

### View Running Apps
```bash
modal app list
```

### Stop Deployment
```bash
modal app stop image-caption-api
```

## Troubleshooting

### Issue: "Dataset files not found"
**Solution:** Run `modal run deploy.py::upload_dataset` before training or deploying.

### Issue: "Model not found"
**Solution:** Run `modal run deploy.py` to regenerate checkpoints; `/models` will reflect whatever lives in `/data/models`.

### Issue: CORS errors in browser
**Solution:** `fastapi_app` adds permissive CORS middleware. Redeploy if you recently changed origins, and verify the Lovable app is calling the Modal URL directly (HTTPS).

### Issue: Slow first inference
**Solution:** The first call loads the model + VGG encoder into memory. Subsequent requests are cached and much faster.

## Cost Optimization

- Modal charges based on compute time
- Models are loaded on-demand and cached
- Use `timeout` settings to limit long-running requests
- Consider using smaller GPU instances for inference (A10G instead of T4)

## Next Steps

1. ✅ Deploy the API to Modal
2. ✅ Test endpoints with cURL or Postman
3. ✅ Integrate with your Lovable app
4. ✅ Monitor usage and costs in Modal dashboard
5. ✅ Fine-tune models based on performance

## Support

For issues or questions:
- Modal Documentation: https://modal.com/docs
- Modal Discord: https://discord.gg/modal
- Project Repository: [Your GitHub URL]
