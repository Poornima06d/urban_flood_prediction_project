CREATE DATABASE IF NOT EXISTS urban_flood;
USE urban_flood;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    district VARCHAR(100) NOT NULL,
    probability DECIMAL(6,4) NOT NULL,
    risk_level ENUM('Low','Medium','High') NOT NULL,
    rainfall_mm DECIMAL(10,2),
    temperature_c DECIMAL(6,2),
    humidity_pct DECIMAL(6,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    district VARCHAR(100) NOT NULL,
    risk_level ENUM('Low','Medium','High') NOT NULL,
    message VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
