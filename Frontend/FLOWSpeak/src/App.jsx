import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './components/Home';
import StutterHelp from './components/StutterHelp';
import Therapy from './components/Therapy';

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
            <Link to="/therapy" className="nav-link">Therapy</Link>
          </div>
        </nav>
        
        <main className='main'>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/stutter-help" element={<StutterHelp />} />
            <Route path="/therapy" element={<Therapy />} /> 
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;