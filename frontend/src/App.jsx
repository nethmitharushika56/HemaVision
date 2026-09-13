import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      setError("");

      const response = await axios.post(
        "http://127.0.0.1:8010/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data);
    } catch (err) {
      console.error(err);

      setError(
        "Prediction failed. Make sure the FastAPI backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>HemaVision</h1>

        <p>
          Explainable Blood Cell Classification
        </p>
      </header>

      <main className="container">
        <section className="upload-card">
          <h2>Upload Microscope Image</h2>

          <p className="description">
            Upload a blood-cell microscope image to classify it as
            Eosinophil, Lymphocyte, Monocyte, or Neutrophil.
          </p>

          <label className="file-label">
            Choose Image

            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />
          </label>

          {preview && (
            <div className="preview-container">
              <img
                src={preview}
                alt="Blood cell preview"
                className="preview-image"
              />
            </div>
          )}

          <button
            className="analyze-button"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Analyze Blood Cell"}
          </button>

          {error && (
            <p className="error">
              {error}
            </p>
          )}
        </section>

        {result && (
          <section className="result-card">
            <h2>Prediction Result</h2>

            <div className="prediction-main">
              <span className="prediction-label">
                Predicted Cell
              </span>

              <h3>
                {result.predicted_class}
              </h3>

              <div className="confidence">
                {(result.confidence * 100).toFixed(2)}%
                Confidence
              </div>
            </div>

            <div className="probability-section">
              <h3>Class Probabilities</h3>

              {Object.entries(result.probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([className, probability]) => (
                  <div
                    className="probability-item"
                    key={className}
                  >
                    <div className="probability-header">
                      <span>{className}</span>

                      <span>
                        {(probability * 100).toFixed(2)}%
                      </span>
                    </div>

                    <div className="progress-background">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${probability * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
            </div>

            <p className="disclaimer">
              {result.disclaimer}
            </p>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;