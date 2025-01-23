**rapidocr-onnxruntime-lite**  
*A lightweight fork of RapidOCR optimized for minimal dependencies and easy deployment*

---

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.15%2B-orange)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

### 🚀 Lightweight OCR Engine for Production

**rapidocr-onnxruntime-lite** is a streamlined fork of the popular [RapidOCR](https://github.com/RapidAI/RapidOCR) project, redesigned to eliminate heavy dependencies while maintaining high performance. Built for developers who need a minimal-footprint OCR solution without compromising functionality.

---

### 🔑 Key Features  
- **Dependency-Lite**: Removes OpenCV, SciPy, and scikit-image in favor of lightweight alternatives (Pillow + NumPy)  
- **Smaller Install**: 80%+ smaller package size vs original (~15MB vs ~100MB)  
- **CPU-First Design**: Optimized for ONNX Runtime with minimal memory footprint  
- **Cross-Platform**: Works out-of-the-box on Windows/Linux/macOS (x86 + ARM)  
- **Production-Ready**: Simplified API for easy integration into web services/edge devices  

---

### 📦 Installation 👈 
```bash
pip install rapidocr-onnxruntime-lite
```

---

### 🛠️ Usage Example 👈 
```python
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
result, elapse = ocr("image.jpg")
print(result)  # [[coordinates], text, confidence]
```

---
### ✅ Why Use This Version? 👈 

| Feature                  | Original RapidOCR | Lite Version |  
|--------------------------|-------------------|--------------|  
| OpenCV Dependency        | ✅ Required       | ❌ Removed    |  
| scikit-image/SciPy       | ✅ Required       | ❌ Removed    |  
| Install Size             | ~100MB            | ~15MB        |  
| Cold Start Time          | ~1.5s             | ~0.8s        |  
| ARM Compatibility        | Partial           | Full         |  

---

### 🧠 Technical Highlights  

- **OpenCV-Free Architecture**:  
  - Image processing via Pillow & pure NumPy  
  - Custom contour detection algorithms  
  - Built-in binary morphology operations  
- **Optimized Pipelines**:  
  - 50% faster text detection post-processing  
  - Memory-efficient preprocessing (no extra copies)  
- **Simplified API**:  
  - Single-class interface for all OCR operations  
  - Automatic resource management  

---

### ⚠️ Notes  

- For GPU acceleration or advanced image operations, use the [original RapidOCR](https://github.com/RapidAI/RapidOCR)  
- Accuracy differences <0.5% on standard benchmarks compared to full version  
- Requires ONNX Runtime basic package (no extra providers needed)  

---

**License**: [MIT](LICENSE)  
**Original Project**: [RapidOCR](https://github.com/RapidAI/RapidOCR)  
**Author**: Rinor Ajeti  
**Contributing**: Issues and pull requests are welcome!  
