import { useState, useEffect } from 'react';
import { getLatestHeatmap } from '../services/api';

const HeatmapView = ({ cameraId }) => {
  const [heatmap, setHeatmap] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchHeatmap = async () => {
    try {
      const data = await getLatestHeatmap(cameraId);
      setHeatmap(data);
      setError(null);
    } catch (err) {
      if (err.response?.status === 404) {
        setError('No heatmap available yet. Waiting for first generation...');
      } else {
        setError('Failed to fetch heatmap');
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHeatmap();
    const interval = setInterval(fetchHeatmap, 60000);
    return () => clearInterval(interval);
  }, [cameraId]);

  return (
    <div className="card shadow-sm">
      <div className="card-body">
        <h5 className="card-title mb-3">
          <i className="bi bi-thermometer-half me-2"></i>
          Activity Heatmap
        </h5>
        
        {loading ? (
          <div className="text-center py-5">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        ) : error ? (
          <div className="alert alert-info" role="alert">
            <i className="bi bi-info-circle me-2"></i>
            {error}
          </div>
        ) : heatmap ? (
          <>
            <div className="position-relative">
              <img 
                src={heatmap.url} 
                alt="Activity Heatmap" 
                className="img-fluid rounded"
                style={{ width: '100%', maxHeight: '500px', objectFit: 'contain' }}
              />
            </div>
            <div className="mt-3 d-flex justify-content-between align-items-center text-muted small">
              <span>
                <i className="bi bi-clock me-1"></i>
                Generated: {new Date(heatmap.timestamp).toLocaleString()}
              </span>
              <span>
                <i className="bi bi-speedometer2 me-1"></i>
                {heatmap.processing_time?.toFixed(2)}s
              </span>
            </div>
            <div className="mt-2">
              <small className="text-muted">
                <i className="bi bi-arrow-repeat me-1"></i>
                Updates every 5 minutes
              </small>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
};

export default HeatmapView;
