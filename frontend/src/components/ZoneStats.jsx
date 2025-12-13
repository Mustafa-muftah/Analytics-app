import { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { getZoneStats } from '../services/api';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const ZoneStats = () => {
  const [zoneData, setZoneData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchZoneStats = async () => {
    try {
      const data = await getZoneStats();
      setZoneData(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch zone statistics');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchZoneStats();
    const interval = setInterval(fetchZoneStats, 15000);
    return () => clearInterval(interval);
  }, []);

  const chartData = zoneData ? {
    labels: zoneData.zones.map(z => z.zone),
    datasets: [
      {
        label: 'Total Visitors',
        data: zoneData.zones.map(z => z.total_visitors),
        backgroundColor: [
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 99, 132, 0.8)',
          'rgba(255, 206, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(153, 102, 255, 0.8)',
        ],
        borderWidth: 1
      }
    ]
  } : null;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: 'y',
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Visitors by Zone',
        font: { size: 14 }
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const zone = zoneData.zones[context.dataIndex];
            return `${context.raw} visitors (${zone.percentage}%)`;
          }
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Number of Visitors'
        }
      }
    }
  };

  return (
    <div className="card shadow-sm">
      <div className="card-body">
        <h5 className="card-title mb-3">
          <i className="bi bi-pin-map me-2"></i>
          Zone Analytics
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
        ) : zoneData && zoneData.zones.length > 0 ? (
          <>
            {/* Chart */}
            <div style={{ height: '200px' }} className="mb-3">
              <Bar data={chartData} options={chartOptions} />
            </div>

            {/* Zone Cards */}
            <div className="row g-2 mt-2">
              {zoneData.zones.map((zone, idx) => (
                <div key={zone.camera_id} className="col-6">
                  <div className={`p-2 rounded bg-opacity-10 ${
                    idx === 0 ? 'bg-primary' : 
                    idx === 1 ? 'bg-danger' : 
                    idx === 2 ? 'bg-warning' : 'bg-success'
                  }`}>
                    <div className="fw-bold text-capitalize">{zone.zone}</div>
                    <div className="small text-muted">
                      {zone.total_visitors} visitors ({zone.percentage}%)
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Total */}
            <div className="text-center mt-3 pt-2 border-top">
              <small className="text-muted">
                <i className="bi bi-people me-1"></i>
                Total across all zones: <strong>{zoneData.total_visitors}</strong>
              </small>
            </div>
          </>
        ) : (
          <div className="text-center py-4 text-muted">
            <i className="bi bi-pin-map fs-1 mb-2 d-block"></i>
            <p className="mb-0">No zone data available yet</p>
            <small>Data will appear after processing</small>
          </div>
        )}
      </div>
    </div>
  );
};

export default ZoneStats;
