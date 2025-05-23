#include <WiFi.h>
#include <WebServer.h>

// WiFi credentials
const char* ssid = "Innovation_Foundry";
const char* password = "@Innovate22";

// Create a web server on port 80
WebServer server(80);

// Global variable for motor speed (0 to 255)
int currentSpeed = 255;

// Motor pin definitions
const int enA = 19;  // PWM speed control Motor A
const int in1 = 18;  // Direction Motor A
const int in2 = 5;
const int in3 = 17;  // Direction Motor B
const int in4 = 16;
const int enB = 4;   // PWM speed control Motor B

// HTML page with buttons and a speed slider
const char* htmlPage = R"=====(
<!DOCTYPE html>
<html>
<head>
  <title>ESP32 Controller</title>
  <style>
    body { text-align: center; font-family: Arial, sans-serif; }
    button { background-color: slateblue; color: white; border: none; width: 80px; height: 30px; margin: 5px; }
    button:hover { background-color: darkslateblue; }
    button:active { background-color: mediumslateblue; }
    input[type=range] { width: 200px; }
  </style>
</head>
<body>
  <h1>ESP32 Controller</h1>
  <p>
    <button onclick="location.href='/go/forward'">FORWARD</button>
    <button onclick="location.href='/go/backward'">BACKWARD</button>
  </p>
  <p>
    <label for="speedSlider">Speed: <span id="speedVal">255</span></label><br>
    <input type="range" id="speedSlider" min="0" max="255" value="255" oninput="updateSpeed(this.value)">
  </p>
  <script>
    function updateSpeed(val) {
      document.getElementById('speedVal').innerText = val;
      var xhr = new XMLHttpRequest();
      xhr.open("GET", "/set/speed?value=" + val, true);
      xhr.send();
    }
  </script>
</body>
</html>
)=====";

// -------------------------
// Function Prototypes
// -------------------------
void handleRoot();
void handleGoForward();
void handleGoBackward();
void handleSetSpeed();
void handleMoveCommand();  // <-- new
void goForward();
void goBackward();
void setSpeed(int motorNumber, int speed);
void setForward(int motorNumber);
void setBackward(int motorNumber);
void setOff(int motorNumber);

// -------------------------
// Setup
// -------------------------
void setup() {
  Serial.begin(115200);

  // Initialize motor control pins
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  // Ensure motors are off initially
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Define HTTP endpoints
  server.on("/", HTTP_GET, handleRoot);
  server.on("/go/forward", HTTP_GET, handleGoForward);
  server.on("/go/backward", HTTP_GET, handleGoBackward);
  server.on("/set/speed", HTTP_GET, handleSetSpeed);
  
  // New endpoint to handle move commands via direction parameter
  server.on("/move", HTTP_GET, handleMoveCommand);

  // Start the server
  server.begin();
  Serial.println("HTTP server started");
}

// -------------------------
// Main Loop
// -------------------------
void loop() {
  server.handleClient();
}

// -------------------------
// HTTP Handlers
// -------------------------

// Serve the main page
void handleRoot() {
  server.send(200, "text/html", htmlPage);
}

// Handle forward command
void handleGoForward() {
  goForward();
  handleRoot();
}

// Handle backward command
void handleGoBackward() {
  goBackward();
  handleRoot();
}

// Handle speed setting: expects a query parameter "value"
void handleSetSpeed() {
  if (server.hasArg("value")) {
    currentSpeed = server.arg("value").toInt();
    Serial.print("Speed set to: ");
    Serial.println(currentSpeed);
    server.send(200, "text/plain", "Speed set");
  } else {
    server.send(400, "text/plain", "Bad Request");
  }
}

// Handle move command (new)
void handleMoveCommand() {
  if (server.hasArg("direction")) {
    String direction = server.arg("direction");
    direction.toLowerCase(); // make sure it's lower-case for comparison
    if (direction == "forward") {
      goForward();
      server.send(200, "text/plain", "Moving forward");
    } else if (direction == "backward") {
      goBackward();
      server.send(200, "text/plain", "Moving backward");
    } else {
      server.send(400, "text/plain", "Unknown direction");
    }
  } else {
    server.send(400, "text/plain", "Missing 'direction' query parameter");
  }
}

// -------------------------
// Motor Control Functions
// -------------------------
void goForward(){
  // Use currentSpeed as set via the web or default (255)
  setSpeed(0, currentSpeed);
  setSpeed(1, currentSpeed);

  // Turn on motor A & B
  setForward(0);
  setForward(1);

  delay(1000);
  // Turn off motors
  setOff(0);
  setOff(1);
}

void goBackward(){
  // Use currentSpeed as set via the web or default (255)
  setSpeed(0, currentSpeed);
  setSpeed(1, currentSpeed);

  // Turn on motor A & B
  //setBackward(0);
  setForward(0);
  setBackward(1);

  delay(1000);
  // Turn off motors
  setOff(0);
  setOff(1);
}

void setSpeed(int motorNumber, int speed){
  if(motorNumber == 0){
    analogWrite(enA, speed);
  } else if(motorNumber == 1){
    analogWrite(enB, speed);
  }
}

//Sets motor to the "forward" position
void setForward(int motorNumber){
  if(motorNumber == 0){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else if(motorNumber == 1){
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
  }
}

//Sets motor to the "backward" position
void setBackward(int motorNumber){
  if(motorNumber == 0){
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  } else if(motorNumber == 1){
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
  }
}

//sets motor to the "off" position
void setOff(int motorNumber){
  if(motorNumber == 0){
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  } else if(motorNumber == 1){
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
  }
}

