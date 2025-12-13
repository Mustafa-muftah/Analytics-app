import { useState } from 'react';
import { getReportSummary } from '../services/api';

const ReportGenerator = ({ cameraId }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const formatTime = (seconds) => {
    if (!seconds) return '0s';
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };

  const generatePDF = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await getReportSummary(cameraId);
      
      // Create printable HTML content
      const content = `
        <!DOCTYPE html>
        <html>
        <head>
          <title>Retail Analytics Report</title>
          <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
              font-family: 'Segoe UI', Arial, sans-serif; 
              padding: 40px; 
              color: #333;
              line-height: 1.6;
            }
            .header { 
              text-align: center; 
              margin-bottom: 40px; 
              padding-bottom: 20px;
              border-bottom: 3px solid #0d6efd;
            }
            .header h1 { 
              color: #0d6efd; 
              font-size: 28px;
              margin-bottom: 10px;
            }
            .header .subtitle { 
              color: #666; 
              font-size: 14px;
            }
            .section { 
              margin-bottom: 30px; 
              page-break-inside: avoid;
            }
            .section-title { 
              font-size: 18px; 
              color: #0d6efd; 
              margin-bottom: 15px;
              padding-bottom: 8px;
              border-bottom: 1px solid #dee2e6;
            }
            .stats-grid { 
              display: grid; 
              grid-template-columns: repeat(2, 1fr); 
              gap: 20px;
            }
            .stat-card { 
              background: #f8f9fa; 
              padding: 20px; 
              border-radius: 8px;
              text-align: center;
            }
            .stat-value { 
              font-size: 36px; 
              font-weight: bold; 
              color: #0d6efd;
            }
            .stat-label { 
              color: #666; 
              font-size: 14px;
              margin-top: 5px;
            }
            .gender-bar {
              display: flex;
              height: 30px;
              border-radius: 15px;
              overflow: hidden;
              margin: 15px 0;
            }
            .gender-male {
              background: #0d6efd;
              display: flex;
              align-items: center;
              justify-content: center;
              color: white;
              font-weight: bold;
            }
            .gender-female {
              background: #dc3545;
              display: flex;
              align-items: center;
              justify-content: center;
              color: white;
              font-weight: bold;
            }
            .peak-hours {
              display: flex;
              gap: 10px;
              flex-wrap: wrap;
            }
            .peak-hour {
              background: #e7f1ff;
              padding: 10px 20px;
              border-radius: 20px;
              font-weight: bold;
              color: #0d6efd;
            }
            .heatmap-container {
              text-align: center;
              margin-top: 20px;
            }
            .heatmap-container img {
              max-width: 100%;
              border-radius: 8px;
              box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .footer {
              margin-top: 40px;
              padding-top: 20px;
              border-top: 1px solid #dee2e6;
              text-align: center;
              color: #666;
              font-size: 12px;
            }
            .dwell-time-box {
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              color: white;
              padding: 25px;
              border-radius: 12px;
              text-align: center;
            }
            .dwell-time-value {
              font-size: 42px;
              font-weight: bold;
            }
            .dwell-time-range {
              margin-top: 10px;
              opacity: 0.9;
            }
            @media print {
              body { padding: 20px; }
              .section { page-break-inside: avoid; }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>📊 Retail Video Analytics Report</h1>
            <div class="subtitle">
              ${data.camera_id} | ${data.period} | Generated: ${new Date(data.generated_at).toLocaleString()}
            </div>
          </div>

          <div class="section">
            <h2 class="section-title">👥 Visitor Statistics</h2>
            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-value">${data.visitors.unique_tracked || data.visitors.total}</div>
                <div class="stat-label">Unique Visitors</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">${data.visitors.peak}</div>
                <div class="stat-label">Peak Concurrent</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">${data.visitors.average_per_hour}</div>
                <div class="stat-label">Average per Hour</div>
              </div>
              <div class="stat-card">
                <div class="stat-value">${data.visitors.total}</div>
                <div class="stat-label">Total Detections</div>
              </div>
            </div>
          </div>

          <div class="section">
            <h2 class="section-title">⏱️ Average Time in Store</h2>
            <div class="dwell-time-box">
              <div class="dwell-time-value">${data.dwell_time.average_formatted}</div>
              <div class="dwell-time-range">
                Range: ${formatTime(data.dwell_time.min_seconds)} - ${formatTime(data.dwell_time.max_seconds)}
              </div>
            </div>
          </div>

          <div class="section">
            <h2 class="section-title">👫 Gender Distribution</h2>
            <div class="gender-bar">
              <div class="gender-male" style="width: ${data.gender.male_percentage}%">
                ♂ ${data.gender.male_percentage}%
              </div>
              <div class="gender-female" style="width: ${data.gender.female_percentage}%">
                ♀ ${data.gender.female_percentage}%
              </div>
            </div>
            <div style="display: flex; justify-content: space-between; color: #666;">
              <span>Male: ${data.gender.male} visitors</span>
              <span>Female: ${data.gender.female} visitors</span>
            </div>
          </div>

          <div class="section">
            <h2 class="section-title">🕐 Peak Hours</h2>
            <div class="peak-hours">
              ${data.peak_hours.map(h => `
                <div class="peak-hour">${h.hour}:00 (${h.avg_count} avg)</div>
              `).join('')}
            </div>
          </div>

          ${data.heatmap_url ? `
          <div class="section">
            <h2 class="section-title">🔥 Traffic Heatmap</h2>
            <div class="heatmap-container">
              <img src="${data.heatmap_url}" alt="Traffic Heatmap" />
            </div>
          </div>
          ` : ''}

          <div class="footer">
            <p>Retail Video Analytics System | Powered by AI Computer Vision</p>
            <p>This report was automatically generated based on video analysis data.</p>
          </div>
        </body>
        </html>
      `;

      // Open print dialog
      const printWindow = window.open('', '_blank');
      printWindow.document.write(content);
      printWindow.document.close();
      
      // Wait for images to load then print
      setTimeout(() => {
        printWindow.print();
      }, 500);

    } catch (err) {
      setError('Failed to generate report');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card shadow-sm">
      <div className="card-body">
        <h5 className="card-title mb-3">
          <i className="bi bi-file-earmark-pdf me-2"></i>
          Export Report
        </h5>
        
        <p className="text-muted small mb-3">
          Generate a comprehensive PDF report with all analytics data from the last 24 hours.
        </p>

        {error && (
          <div className="alert alert-danger py-2 mb-3">
            <i className="bi bi-exclamation-circle me-2"></i>
            {error}
          </div>
        )}

        <button
          className="btn btn-outline-primary w-100"
          onClick={generatePDF}
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="spinner-border spinner-border-sm me-2"></span>
              Generating...
            </>
          ) : (
            <>
              <i className="bi bi-download me-2"></i>
              Generate PDF Report
            </>
          )}
        </button>

        <div className="mt-3 small text-muted">
          <i className="bi bi-info-circle me-1"></i>
          Report includes: Visitors, Gender, Dwell Time, Peak Hours, Heatmap
        </div>
      </div>
    </div>
  );
};

export default ReportGenerator;
