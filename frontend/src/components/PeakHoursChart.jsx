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
import { getPeakHours } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const PeakHoursChart = ({ cameraId }) => {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchPeakHours = async () => {
    try {
      const data = await getPeakHours(cameraId);
      
      const labels = data.map(item => `${item.hour}:00`);
      const avgCounts = data.map(item => item.avg_count);
      const maxCounts = data.map(item => item.max_count);

      setChartData({
        labels,
        datasets: [
          {
            label: 'Average Count',
            data: avgCounts,
            backgroundColor: 'rgba(54, 162, 235, 0.6)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          },
          {
            label: 'Peak Count',
            data: maxCounts,
            backgroundColor: 'rgba(255, 99, 132, 0.6)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
          }
        ]
      });
      setError(null);
    } catch (err) {
      setError('Failed to fetch peak hours data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPeakHours();
    const interval = setInterval(fetchPeakHours, 30000);
    return () => clearInterval(interval);
  }, [cameraId]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top'
      },
      title: {
        display: true,
        text: 'Peak Hours Analysis (Last 24 Hours)',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)} people`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Number of People'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Hour of Day'
        }
      }
    }
  };

  return (
    <div className="card shadow-sm">
      <div className="card-body">
        {loading ? (
          <div className="text-center py-5">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        ) : error ? (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        ) : chartData ? (
          <div style={{ height: '400px' }}>
            <Bar data={chartData} options={options} />
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default PeakHoursChart;
