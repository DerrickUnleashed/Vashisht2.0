import React, { useState } from 'react';
import { motion } from 'framer-motion';
import './Home.css';

const Home = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const mailtoLink = `mailto:support@flowspeak.ai?subject=Contact Form Submission&body=Name: ${formData.name}%0AEmail: ${formData.email}%0AMessage: ${formData.message}`;
    window.location.href = mailtoLink; // Open the mail client with pre-filled information
    setIsSubmitted(true);
    setFormData({ name: '', email: '', message: '' });
    setTimeout(() => setIsSubmitted(false), 5000);
  };


  return (
    <div className="home-container">
      <motion.div 
        className="hero-section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <img src="../../public/fslogov1.png" alt="" className="logo" />
        <br /><br /><br />
        <h1 className="title">
          <span className="highlight">FLOW</span>speak
        </h1>
        <h2 className="subtitle">Personalized Stutter Detection and Helper Model</h2>
        
        <motion.div 
          className="animated-wave"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <svg viewBox="0 0 1440 320" className="wave-svg">
            <path fill="#4f46e5" fillOpacity="0.4" d="M0,224L48,213.3C96,203,192,181,288,181.3C384,181,480,203,576,218.7C672,235,768,245,864,234.7C960,224,1056,192,1152,186.7C1248,181,1344,203,1392,213.3L1440,224L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
            <path fill="#4f46e5" fillOpacity="0.2" d="M0,160L48,176C96,192,192,224,288,234.7C384,245,480,235,576,213.3C672,192,768,160,864,165.3C960,171,1056,213,1152,218.7C1248,224,1344,192,1392,176L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
          </svg>
        </motion.div>
      </motion.div>

      <section className="workflow-section">
        <motion.h2 
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
        >
          How FLOWspeak Works
        </motion.h2>
        
        <div className="workflow-steps">
          {[
            { icon: "🎤", title: "Speech Capture", desc: "Real-time audio capture from your device" },
            { icon: "🔍", title: "Stutter Detection", desc: "Advanced ML algorithms identify stutter patterns" },
            { icon: "🧩", title: "Classification", desc: "Categorization of different stutter types" },
            { icon: "✨", title: "Correction", desc: "Smart suggestions and autocomplete to aid fluency" }
          ].map((step, index) => (
            <motion.div 
              className="workflow-step"
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
              viewport={{ once: true }}
            >
              <div className="step-icon">{step.icon}</div>
              <h3>{step.title}</h3>
              <p>{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="mission-section">
        <motion.div 
          className="mission-content"
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <h2>Our Mission</h2>
          <p>
            FLOWspeak was created to democratize speech therapy and support for those with speech disfluencies. Many people 
            around the world wish to develop their speech skills but lack access to professional help due to geographical, 
            financial, or social barriers.
          </p>
          <p>
            Our AI-powered platform provides real-time detection, classification, and assistance for people with stuttering 
            issues, helping them practice and improve in a supportive, private environment. We believe technology should 
            empower everyone to communicate with confidence.
          </p>
          <motion.button 
            className="cta-button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => window.location.href = '/stutter-help'}
          >
            Try FLOWspeak Now
          </motion.button>
        </motion.div>
      </section>

      <section className="contact-section" id="contact">
        <h2>Contact Us</h2>
        <div className="contact-container">
          <div className="form-description">
            <h3>We'd Love to Hear From You</h3>
            <p>
              Have questions, suggestions, or feedback about FLOWspeak? 
              We're constantly improving our platform based on user insights.
            </p>
            <div className="contact-info">
              <div className="contact-item">
                <span className="contact-icon">📧</span>
                <span>support@flowspeak.ai</span>
              </div>
              <div className="contact-item">
                <span className="contact-icon">🌐</span>
                <span>www.flowspeak.ai</span>
              </div>
            </div>
          </div>
          
          <motion.form 
            className="contact-form"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            onSubmit={handleSubmit}
          >
            {isSubmitted ? (
              <div className="success-message">
                <motion.div 
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 260, damping: 20 }}
                >
                  <span className="success-icon">✓</span>
                  <p>Thank you for your message! We'll get back to you soon.</p>
                </motion.div>
              </div>
            ) : (
              <>
                <div className="form-group">
                  <label htmlFor="name">Name</label>
                  <input 
                    type="text" 
                    id="name" 
                    name="name" 
                    value={formData.name}
                    onChange={handleChange}
                    required 
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="email">Email</label>
                  <input 
                    type="email" 
                    id="email" 
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    required 
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="message">Message</label>
                  <textarea 
                    id="message" 
                    name="message" 
                    rows="4"
                    value={formData.message}
                    onChange={handleChange}
                    required
                  ></textarea>
                </div>
                <motion.button 
                  type="submit" 
                  className="submit-button"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Send Message
                </motion.button>
              </>
            )}
          </motion.form>
        </div>
      </section>

      
    </div>
  );
};

export default Home;
