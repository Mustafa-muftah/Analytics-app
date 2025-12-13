import { useState, useEffect } from 'react';
import { getCameras } from '../services/api';

const CameraSelector = ({ selectedCamera, onCameraChange }) => {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const data = await getCameras();
        setCameras(data);
      } catch (err) {
        console.error('Failed to fetch cameras:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchCameras();
  }, []);

  // Don't show selector if only one camera
  if (!loading && cameras.length <= 1) {
    return null;
  }

  return (
    <div className="mb-3">
      <div className="d-flex align-items-center gap-2">
        <i className="bi bi-camera-video text-primary"></i>
        <select
          className="form-select form-select-sm"
          value={selectedCamera || ''}
          onChange={(e) => onCameraChange(e.target.value || null)}
          disabled={loading}
          style={{ maxWidth: '200px' }}
        >
          <option value="">All Cameras</option>
          {cameras.map((cam) => (
            <option key={cam.id} value={cam.id}>
              {cam.name} ({cam.zone})
            </option>
          ))}
        </select>
        {loading && (
          <div className="spinner-border spinner-border-sm text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraSelector;
