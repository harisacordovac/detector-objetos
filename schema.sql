CREATE DATABASE IF NOT EXISTS detecciones_db CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE detecciones_db;

CREATE TABLE IF NOT EXISTS detecciones (
  id INT AUTO_INCREMENT PRIMARY KEY,
  objeto VARCHAR(64),
  forma ENUM('rectangulo','cuadrado','circulo'),
  color VARCHAR(32),
  fecha_hora DATETIME,
  imagen_url VARCHAR(255)
);
