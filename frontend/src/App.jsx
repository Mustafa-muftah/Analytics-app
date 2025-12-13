import { useState, useEffect } from 'react';
import PeopleCounter from './components/PeopleCounter';
import PeakHoursChart from './components/PeakHoursChart';
import HeatmapView from './components/HeatmapView';
import GenderStats from './components/GenderStats';
import ZoneStats from './components/ZoneStats';
import DwellTimeStats from './components/DwellTimeStats';
import CameraSelector from './components/CameraSelector';
import BatchProcessor from './components/BatchProcessor';
import ReportGenerator from './components/ReportGenerator';
import { getStats, getCameras } from './services/api';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';

function App() {
  const [stats, setStats] = useState(null);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [showBatchProcessor, setShowBatchProcessor] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsData, camerasData] = await Promise.all([
          getStats(),
          getCameras()
        ]);
        setStats(statsData);
        setHasMultipleCameras(camerasData.length > 1);
        
        // Check if any camera is in batch mode
        const hasBatchCamera = camerasData.some(c => c.mode === 'batch' || c.mode === 'auto');
        setShowBatchProcessor(hasBatchCamera);
      } catch (err) {
        console.error('Failed to fetch data:', err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-vh-100" style={{ backgroundColor: '#f0f2f5' }}>
      {/* Header */}
      <nav className="navbar navbar-expand-lg navbar-dark shadow-sm" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <div className="container-fluid px-4">
          <span className="navbar-brand mb-0 h1 d-flex align-items-center">
            <i className="bi bi-bar-chart-line-fill me-2 fs-4"></i>
            <span>Retail Analytics</span>
          </span>
          
          {/* Navigation Tabs */}
          <div className="d-none d-md-flex">
            <ul className="navbar-nav flex-row gap-1">
              <li className="nav-item">
                <button 
                  className={`nav-link btn btn-link ${activeTab === 'dashboard' ? 'active bg-white bg-opacity-25 rounded' : ''}`}
                  onClick={() => setActiveTab('dashboard')}
                >
                  <i className="bi bi-speedometer2 me-1"></i> Dashboard
                </button>
              </li>
              <li className="nav-item">
                <button 
                  className={`nav-link btn btn-link ${activeTab === 'analytics' ? 'active bg-white bg-opacity-25 rounded' : ''}`}
                  onClick={() => setActiveTab('analytics')}
                >
                  <i className="bi bi-graph-up me-1"></i> Analytics
                </button>
              </li>
              <li className="nav-item">
                <button 
                  className={`nav-link btn btn-link ${activeTab === 'reports' ? 'active bg-white bg-opacity-25 rounded' : ''}`}
                  onClick={() => setActiveTab('reports')}
                >
                  <i className="bi bi-file-earmark-text me-1"></i> Reports
                </button>
              </li>
            </ul>
          </div>

          <div className="d-flex align-items-center gap-2">
            {stats && (
              <>
                <span className="badge bg-white bg-opacity-25 text-white">
                  <i className="bi bi-camera me-1"></i>
                  {stats.total_cameras}
                </span>
                <span className={`badge ${stats.camera_status === 'connected' ? 'bg-success' : 'bg-danger'}`}>
                  <i className="bi bi-circle-fill me-1" style={{ fontSize: '8px' }}></i>
                  {stats.camera_status}
                </span>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="container-fluid px-4 py-4">
        {/* Camera Selector */}
        {hasMultipleCameras && (
          <div className="mb-4">
            <CameraSelector
              selectedCamera={selectedCamera}
              onCameraChange={setSelectedCamera}
            />
          </div>
        )}

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="row g-4">
            {/* Top Stats Row */}
            <div className="col-12">
              <div className="row g-3">
                <div className="col-lg-3 col-md-6">
                  <PeopleCounter cameraId={selectedCamera} />
                </div>
                <div className="col-lg-3 col-md-6">
                  <DwellTimeStats cameraId={selectedCamera} />
                </div>
                <div className="col-lg-3 col-md-6">
                  <GenderStats cameraId={selectedCamera} />
                </div>
                <div className="col-lg-3 col-md-6">
                  {showBatchProcessor ? (
                    <BatchProcessor />
                  ) : (
                    <ZoneStats />
                  )}
                </div>
              </div>
            </div>

            {/* Charts Row */}
            <div className="col-lg-8">
              <PeakHoursChart cameraId={selectedCamera} />
            </div>
            <div className="col-lg-4">
              <HeatmapView cameraId={selectedCamera} />
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="row g-4">
            <div className="col-lg-6">
              <PeakHoursChart cameraId={selectedCamera} />
            </div>
            <div className="col-lg-6">
              <div className="row g-4">
                <div className="col-12">
                  <DwellTimeStats cameraId={selectedCamera} />
                </div>
                <div className="col-12">
                  <GenderStats cameraId={selectedCamera} />
                </div>
              </div>
            </div>
            <div className="col-lg-6">
              <ZoneStats />
            </div>
            <div className="col-lg-6">
              <HeatmapView cameraId={selectedCamera} />
            </div>
          </div>
        )}

        {/* Reports Tab */}
        {activeTab === 'reports' && (
          <div className="row g-4">
            <div className="col-lg-4">
              <ReportGenerator cameraId={selectedCamera} />
              
              {showBatchProcessor && (
                <div className="mt-4">
                  <BatchProcessor />
                </div>
              )}

              {/* System Stats */}
              {stats && (
                <div className="card shadow-sm mt-4">
                  <div className="card-body">
                    <h6 className="card-title text-muted mb-3">
                      <i className="bi bi-gear me-2"></i>
                      System Status
                    </h6>
                    <div className="row text-center g-2">
                      <div className="col-6">
                        <div className="p-2 bg-primary bg-opacity-10 rounded">
                          <div className="fs-5 fw-bold text-primary">
                            {stats.total_counts?.toLocaleString() || 0}
                          </div>
                          <small className="text-muted">Records</small>
                        </div>
                      </div>
                      <div className="col-6">
                        <div className="p-2 bg-success bg-opacity-10 rounded">
                          <div className="fs-5 fw-bold text-success">
                            {stats.total_heatmaps || 0}
                          </div>
                          <small className="text-muted">Heatmaps</small>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <div className="col-lg-8">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h5 className="card-title mb-4">
                    <i className="bi bi-info-circle me-2"></i>
                    Report Information
                  </h5>
                  
                  <div className="row g-4">
                    <div className="col-md-6">
                      <h6 className="text-muted mb-3">📊 What's Included</h6>
                      <ul className="list-unstyled">
                        <li className="mb-2"><i className="bi bi-check-circle text-success me-2"></i>Unique visitor count</li>
                        <li className="mb-2"><i className="bi bi-check-circle text-success me-2"></i>Gender distribution</li>
                        <li className="mb-2"><i className="bi bi-check-circle text-success me-2"></i>Average dwell time</li>
                        <li className="mb-2"><i className="bi bi-check-circle text-success me-2"></i>Peak hours analysis</li>
                        <li className="mb-2"><i className="bi bi-check-circle text-success me-2"></i>Traffic heatmap</li>
                        <li className="mb-2"><i className="bi bi-check-circle text-success me-2"></i>Zone statistics</li>
                      </ul>
                    </div>
                    <div className="col-md-6">
                      <h6 className="text-muted mb-3">💡 Tips</h6>
                      <ul className="list-unstyled">
                        <li className="mb-2"><i className="bi bi-lightbulb text-warning me-2"></i>Process video before generating report</li>
                        <li className="mb-2"><i className="bi bi-lightbulb text-warning me-2"></i>Use batch mode for video files</li>
                        <li className="mb-2"><i className="bi bi-lightbulb text-warning me-2"></i>Reports cover last 24 hours</li>
                        <li className="mb-2"><i className="bi bi-lightbulb text-warning me-2"></i>Print to PDF from browser</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Preview Cards */}
              <div className="row g-3 mt-2">
                <div className="col-md-4">
                  <div className="card h-100 shadow-sm border-0" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
                    <div className="card-body text-white text-center py-4">
                      <i className="bi bi-people fs-1 mb-2"></i>
                      <h6>Visitor Analytics</h6>
                      <small className="opacity-75">Track unique visitors accurately</small>
                    </div>
                  </div>
                </div>
                <div className="col-md-4">
                  <div className="card h-100 shadow-sm border-0" style={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' }}>
                    <div className="card-body text-white text-center py-4">
                      <i className="bi bi-clock-history fs-1 mb-2"></i>
                      <h6>Dwell Time</h6>
                      <small className="opacity-75">Measure engagement duration</small>
                    </div>
                  </div>
                </div>
                <div className="col-md-4">
                  <div className="card h-100 shadow-sm border-0" style={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }}>
                    <div className="card-body text-white text-center py-4">
                      <i className="bi bi-fire fs-1 mb-2"></i>
                      <h6>Heatmaps</h6>
                      <small className="opacity-75">Visualize traffic patterns</small>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="text-center text-muted py-4 mt-auto border-top bg-white">
        <small>
          <i className="bi bi-shield-check me-1"></i>
          Retail Video Analytics v2.0 | AI-Powered Insights
        </small>
      </footer>
    </div>
  );
}

export default App;
