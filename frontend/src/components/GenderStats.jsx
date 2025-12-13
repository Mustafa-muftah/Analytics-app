import { useState, useEffect } from 'react';
import { Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend
} from 'chart.js';
import { getGenderStats } from '../services/api';

ChartJS.register(ArcElement, Tooltip, Legend);

const GenderStats = ({ cameraId }) => {
  const [genderData, setGenderData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchGenderStats = async () => {
    try {
      const data = await getGenderStats(cameraId);
      setGenderData(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch gender statistics');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGenderStats();
    const interval = setInterval(fetchGenderStats, 10000);
    return () => clearInterval(interval);
  }, [cameraId]);

  const pieChartData = genderData ? {
    labels: ['Male', 'Female'],
    datasets: [
      {
        data: [genderData.total.male, genderData.total.female],
        backgroundColor: [
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 99, 132, 0.8)'
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(255, 99, 132, 1)'
        ],
        borderWidth: 2
      }
    ]
  } : null;

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 20,
          font: {
            size: 14
          }
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = ((context.raw / total) * 100).toFixed(1);
            return `${context.label}: ${context.raw} (${percentage}%)`;
          }
        }
      }
    }
  };

  return (
    <div className="card shadow-sm">
      <div className="card-body">
        <h5 className="card-title mb-3">
          <i className="bi bi-gender-ambiguous me-2"></i>
          Gender Distribution
        </h5>

        {loading ? (
          <div className="text-center py-4">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        ) : error ? (
          <div className="alert alert-warning" role="alert">
            <i className="bi bi-exclamation-triangle me-2"></i>
            {error}
          </div>
        ) : genderData && genderData.total.total_detected > 0 ? (
          <>
            {/* Pie Chart */}
            <div style={{ height: '200px' }} className="mb-3">
              <Pie data={pieChartData} options={pieOptions} />
            </div>

            {/* Stats Summary */}
            <div className="row text-center mt-3">
              <div className="col-6">
                <div className="p-2 bg-primary bg-opacity-10 rounded">
                  <i className="bi bi-gender-male text-primary fs-4"></i>
                  <div className="fs-4 fw-bold text-primary">
                    {genderData.percentage.male}%
                  </div>
                  <small className="text-muted">
                    Male ({genderData.total.male})
                  </small>
                </div>
              </div>
              <div className="col-6">
                <div className="p-2 bg-danger bg-opacity-10 rounded">
                  <i className="bi bi-gender-female text-danger fs-4"></i>
                  <div className="fs-4 fw-bold text-danger">
                    {genderData.percentage.female}%
                  </div>
                  <small className="text-muted">
                    Female ({genderData.total.female})
                  </small>
                </div>
              </div>
            </div>

            {/* Total Detected */}
            <div className="text-center mt-3 pt-2 border-top">
              <small className="text-muted">
                <i className="bi bi-person-check me-1"></i>
                Unique visitors analyzed: <strong>{genderData.total.male + genderData.total.female}</strong>
              </small>
            </div>
          </>
        ) : (
          <div className="text-center py-4 text-muted">
            <i className="bi bi-person-bounding-box fs-1 mb-2 d-block"></i>
            <p className="mb-0">No gender data available yet</p>
            <small>Waiting for face detections...</small>
          </div>
        )}
      </div>
    </div>
  );
};

export default GenderStats;
