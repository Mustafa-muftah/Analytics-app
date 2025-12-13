import { useState, useEffect } from 'react';
import { getCurrentCount } from '../services/api';

const PeopleCounter = ({ cameraId }) => {
  const [count, setCount] = useState(0);
  const [byCamera, setByCamera] = useState({});
  const [timestamp, setTimestamp] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchCount = async () => {
    try {
      const data = await getCurrentCount(cameraId);
      setCount(data.count);
      setByCamera(data.by_camera || {});
      setTimestamp(new Date(data.timestamp).toLocaleTimeString());
      setError(null);
    } catch (err) {
      setError('Failed to fetch count');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCount();
    const interval = setInterval(fetchCount, 3000);
    return () => clearInterval(interval);
  }, [cameraId]);

  return (
    <div className="card shadow-sm">
      <div className="card-body text-center">
        <h5 className="card-title text-muted mb-3">
          <i className="bi bi-people-fill me-2"></i>
          Current People Count
        </h5>
        
        {loading ? (
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        ) : error ? (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        ) : (
          <>
            <h1 className="display-1 fw-bold text-primary mb-3">
              {count}
            </h1>
            <p className="text-muted small mb-0">
              <i className="bi bi-clock me-1"></i>
              Last updated: {timestamp}
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default PeopleCounter;
