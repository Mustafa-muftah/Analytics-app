import { useState, useEffect } from 'react';
import { getCameras, startBatchProcessing, getBatchProgress, getProcessingJobs } from '../services/api';

const BatchProcessor = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchCameras();
    fetchJobs();
  }, []);

  useEffect(() => {
    let interval;
    if (processing && selectedCamera) {
      interval = setInterval(async () => {
        try {
          const prog = await getBatchProgress(selectedCamera);
          setProgress(prog);
          
          // Check if completed
          if (prog.progress_percent >= 100) {
            setProcessing(false);
            fetchJobs();
          }
        } catch (err) {
          console.error('Progress fetch error:', err);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [processing, selectedCamera]);

  const fetchCameras = async () => {
    try {
      const data = await getCameras();
      // Filter only batch mode cameras
      const batchCameras = data.filter(c => c.mode === 'batch' || c.mode === 'auto');
      setCameras(batchCameras);
      if (batchCameras.length > 0) {
        setSelectedCamera(batchCameras[0].id);
      }
    } catch (err) {
      console.error('Failed to fetch cameras:', err);
    }
  };

  const fetchJobs = async () => {
    try {
      const data = await getProcessingJobs();
      setJobs(data);
    } catch (err) {
      console.error('Failed to fetch jobs:', err);
    }
  };

  const handleStartProcessing = async () => {
    if (!selectedCamera) return;
    
    setError(null);
    setProcessing(true);
    setProgress({ progress_percent: 0 });
    
    try {
      await startBatchProcessing(selectedCamera);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start processing');
      setProcessing(false);
    }
  };

  const getStatusBadge = (status) => {
    const badges = {
      pending: 'bg-secondary',
      processing: 'bg-primary',
      completed: 'bg-success',
      failed: 'bg-danger'
    };
    return badges[status] || 'bg-secondary';
  };

  return (
    <div className="card shadow-sm">
      <div className="card-body">
        <h5 className="card-title mb-3">
          <i className="bi bi-play-circle me-2"></i>
          Batch Video Processing
        </h5>

        {/* Camera Selection */}
        <div className="mb-3">
          <label className="form-label small text-muted">Select Video Source</label>
          <select
            className="form-select"
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
            disabled={processing}
          >
            {cameras.length === 0 ? (
              <option>No batch cameras configured</option>
            ) : (
              cameras.map(cam => (
                <option key={cam.id} value={cam.id}>
                  {cam.name} - {cam.source}
                </option>
              ))
            )}
          </select>
        </div>

        {/* Progress Bar */}
        {processing && progress && (
          <div className="mb-3">
            <div className="d-flex justify-content-between small mb-1">
              <span>Processing...</span>
              <span>{progress.progress_percent}%</span>
            </div>
            <div className="progress" style={{ height: '20px' }}>
              <div
                className="progress-bar progress-bar-striped progress-bar-animated"
                role="progressbar"
                style={{ width: `${progress.progress_percent}%` }}
              >
                {progress.frame_count} / {progress.total_frames} frames
              </div>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="alert alert-danger py-2 mb-3" role="alert">
            <i className="bi bi-exclamation-circle me-2"></i>
            {error}
          </div>
        )}

        {/* Start Button */}
        <button
          className="btn btn-primary w-100 mb-3"
          onClick={handleStartProcessing}
          disabled={processing || cameras.length === 0}
        >
          {processing ? (
            <>
              <span className="spinner-border spinner-border-sm me-2" role="status"></span>
              Processing...
            </>
          ) : (
            <>
              <i className="bi bi-play-fill me-2"></i>
              Start Batch Processing
            </>
          )}
        </button>

        {/* Recent Jobs */}
        {jobs.length > 0 && (
          <>
            <h6 className="text-muted mb-2">Recent Jobs</h6>
            <div className="list-group list-group-flush small">
              {jobs.slice(0, 5).map(job => (
                <div key={job.id} className="list-group-item px-0 py-2 d-flex justify-content-between align-items-center">
                  <div>
                    <span className="fw-medium">{job.camera_id}</span>
                    <br />
                    <small className="text-muted">
                      {job.started_at ? new Date(job.started_at).toLocaleString() : 'Pending'}
                    </small>
                  </div>
                  <div className="text-end">
                    <span className={`badge ${getStatusBadge(job.status)}`}>
                      {job.status}
                    </span>
                    <br />
                    <small className="text-muted">{job.progress}%</small>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Help Text */}
        <div className="mt-3 p-2 bg-light rounded small text-muted">
          <i className="bi bi-info-circle me-1"></i>
          Batch mode processes video files quickly without real-time delays. 
          Ideal for analyzing recorded footage and generating reports.
        </div>
      </div>
    </div>
  );
};

export default BatchProcessor;
