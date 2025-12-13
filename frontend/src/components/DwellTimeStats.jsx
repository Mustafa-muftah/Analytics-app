import { useState, useEffect } from 'react';
import { getVisitorStats, getDwellTime } from '../services/api';

const DwellTimeStats = ({ cameraId }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStats = async () => {
    try {
      const data = await getVisitorStats(cameraId);
      setStats(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch dwell time statistics');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [cameraId]);

  const formatTime = (seconds) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="card shadow-sm">
      <div className="card-body">
        <h5 className="card-title mb-3">
          <i className="bi bi-stopwatch me-2"></i>
          Visitor Analytics
        </h5>

        {loading ? (
          <div className="text-center py-4">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        ) : error ? (
          <div className="alert alert-warning py-2" role="alert">
            <i className="bi bi-exclamation-triangle me-2"></i>
            {error}
          </div>
        ) : stats ? (
          <>
            {/* Main Stats Grid */}
            <div className="row g-3 mb-3">
              {/* Unique Visitors */}
              <div className="col-6">
                <div className="p-3 bg-primary bg-opacity-10 rounded text-center">
                  <i className="bi bi-person-check text-primary fs-4"></i>
                  <div className="fs-3 fw-bold text-primary">
                    {stats.unique_visitors}
                  </div>
                  <small className="text-muted">Unique Visitors</small>
                </div>
              </div>
              
              {/* Active Now */}
              <div className="col-6">
                <div className="p-3 bg-success bg-opacity-10 rounded text-center">
                  <i className="bi bi-person-walking text-success fs-4"></i>
                  <div className="fs-3 fw-bold text-success">
                    {stats.active_visitors}
                  </div>
                  <small className="text-muted">In Store Now</small>
                </div>
              </div>
            </div>

            {/* Dwell Time */}
            <div className="p-3 bg-info bg-opacity-10 rounded mb-3">
              <div className="d-flex justify-content-between align-items-center">
                <div>
                  <i className="bi bi-clock-history text-info me-2"></i>
                  <span className="text-muted">Avg. Time in Store</span>
                </div>
                <div className="fs-4 fw-bold text-info">
                  {formatTime(stats.avg_dwell_time || 0)}
                </div>
              </div>
              
              {stats.min_dwell_time !== undefined && stats.max_dwell_time !== undefined && (
                <div className="mt-2 small text-muted d-flex justify-content-between">
                  <span>Min: {formatTime(stats.min_dwell_time)}</span>
                  <span>Max: {formatTime(stats.max_dwell_time)}</span>
                </div>
              )}
            </div>

            {/* Gender Breakdown */}
            {stats.gender_breakdown && (
              <div className="border-top pt-3">
                <small className="text-muted d-block mb-2">
                  <i className="bi bi-gender-ambiguous me-1"></i>
                  Gender Breakdown (Unique Visitors)
                </small>
                <div className="d-flex justify-content-around text-center">
                  <div>
                    <i className="bi bi-gender-male text-primary"></i>
                    <div className="fw-bold">{stats.gender_breakdown.male}</div>
                    <small className="text-muted">Male</small>
                  </div>
                  <div>
                    <i className="bi bi-gender-female text-danger"></i>
                    <div className="fw-bold">{stats.gender_breakdown.female}</div>
                    <small className="text-muted">Female</small>
                  </div>
                  <div>
                    <i className="bi bi-question-circle text-secondary"></i>
                    <div className="fw-bold">{stats.gender_breakdown.unknown}</div>
                    <small className="text-muted">Unknown</small>
                  </div>
                </div>
              </div>
            )}

            {/* Info */}
            <div className="mt-3 pt-2 border-top small text-muted">
              <i className="bi bi-info-circle me-1"></i>
              Tracking uses AI to count unique individuals, avoiding double-counting.
            </div>
          </>
        ) : (
          <div className="text-center py-4 text-muted">
            <i className="bi bi-hourglass fs-1 mb-2 d-block"></i>
            <p className="mb-0">No visitor data yet</p>
            <small>Data will appear after processing</small>
          </div>
        )}
      </div>
    </div>
  );
};

export default DwellTimeStats;
