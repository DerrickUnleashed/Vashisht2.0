import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './components/Home';
// import StutterHelp from './components/StutterHelp'; // commented out for now
import './App.css';

const App = () => {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="logo">
            <Link to="/">FLOW<span>speak</span></Link>
          </div>
          <div className="nav-links">
            <Link to="/" className="nav-link">Home</Link>
            <Link to="/stutter-help" className="nav-link">Stutter Help</Link>
          </div>
        </nav>
        
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            {/* <Route path="/stutter-help" element={<StutterHelp />} /> */}
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;