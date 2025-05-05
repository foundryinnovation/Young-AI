import turtle
import sounddevice as sd
import numpy as np
from datetime import datetime
from vosk import Model, KaldiRecognizer
import json
import random
import os

print("Initializing speech recognition...")
print("Note: First time may take a moment to load the model")

# Initialize speech recognition with the small model
model = Model(lang="en-us")  # This will use the default small model
recognizer = KaldiRecognizer(model, 16000)

# Set up the screen
wn = turtle.Screen()
wn.bgcolor("black")
wn.title("Voice-Controlled Turtle Graphics")
wn.setup(1000, 600)

# Available commands dictionary
COMMANDS = {
    "Shapes": ["square", "triangle", "circle", "star", "hexagon", "kid", "orange", "nuked", "house", "sigma"],
    "Colors": ["red", "blue", "green", "yellow", "white", "purple", "orange", "salmon1"],
    "Actions": ["draw [shape]", "color [color]", "background [color]", "clear", "exit"]
}

coolArray = ["red", "orange", "yellow", "green", "blue", "purple", "hotpink", "chocolate", "honeydew", "indianred"]

def select_microphone():
    """Let user select microphone from available devices"""
    devices = sd.query_devices()
    input_devices = []
    print("\nAvailable Input Devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device['name']))
            print(f"{len(input_devices) - 1}: {device['name']}")

    while True:
        try:
            choice = int(input("\nSelect microphone number (0-{}): ".format(len(input_devices) - 1)))
            if 0 <= choice < len(input_devices):
                device_id = input_devices[choice][0]
                print(f"Selected: {input_devices[choice][1]}")
                sd.default.device = device_id
                return device_id
        except ValueError:
            pass
        print("Invalid selection, please try again")


class TurtleGraphics:
    def __init__(self):
        # Create turtles
        self.text_display = turtle.Turtle()
        self.status_display = turtle.Turtle()
        self.history_display = turtle.Turtle()
        self.artist = turtle.Turtle()

        # Initialize turtles
        for t in [self.text_display, self.status_display, self.history_display]:
            t.hideturtle()
            t.penup()

        # Set positions and colors
        self.text_display.color("white")
        self.text_display.goto(-480, 250)

        self.status_display.color("yellow")
        self.status_display.goto(-480, -200)

        self.history_display.color("lightgreen")
        self.history_display.goto(200, 250)

        self.artist.speed(3)
        self.artist.width(2)
        self.artist.color("white")

        self.command_history = []
        self.display_commands()
        self.current_color = "white"

    def display_commands(self):
        """Display available commands on screen"""
        self.text_display.clear()
        self.text_display.write("Available Commands:", font=("Arial", 12, "bold"))
        y = 220
        for category, items in COMMANDS.items():
            y -= 25
            self.text_display.goto(-480, y)
            self.text_display.write(f"{category}:", font=("Arial", 10, "bold"))
            y -= 20
            for item in items:
                self.text_display.goto(-460, y)
                self.text_display.write(f"• {item}", font=("Arial", 10, "normal"))
                y -= 20

    def update_status(self, message):
        """Update the status display"""
        self.status_display.clear()
        self.status_display.write(message, font=("Arial", 12, "normal"))
        print(message)

    def update_history(self, heard_text, was_command=True):
        """Update the command history display"""
        self.command_history.append((datetime.now().strftime("%H:%M:%S"), heard_text, was_command))
        if len(self.command_history) > 10:
            self.command_history.pop(0)

        self.history_display.clear()
        self.history_display.goto(200, 250)
        self.history_display.write("Command History:", font=("Arial", 12, "bold"))
        y = 220
        for time, text, is_command in self.command_history:
            color = "lightgreen" if is_command else "orange"
            self.history_display.color(color)
            self.history_display.goto(200, y)
            self.history_display.write(f"{time} - {text}", font=("Arial", 10, "normal"))
            y -= 20

    def draw_shape(self, shape):
        """Draw the specified shape"""
        shapes = [s.lower() for s in COMMANDS["Shapes"]]
        shape = shape.lower()

        if shape not in shapes:
            self.update_status(f"Unknown shape. Valid shapes: {', '.join(COMMANDS['Shapes'])}")
            return

        self.artist.color(self.current_color)
        if shape == "square":
            for _ in range(4):
                self.artist.forward(100)
                self.artist.right(90)
        elif shape == "triangle":
            for _ in range(3):
                self.artist.forward(100)
                self.artist.right(120)
        elif shape == "nuked":

            self.artist.forward(250)
            self.artist.right(90)
            self.artist.forward(250)
            self.artist.right(90)
            self.artist.forward(250)
            self.artist.right(90)
            self.artist.forward(125)
            self.artist.right(90)
            self.artist.pencolor("white")
            self.artist.forward(63)
            self.artist.fillcolor("red")
            self.artist.begin_fill()
            self.artist.circle(63)
            self.artist.end_fill()
            self.artist.pencolor("white")
            self.artist.forward(63)
            self.artist.pencolor("black")
            self.artist.right(90)
            self.artist.forward(125)


        elif shape == "sigma":


            # Draw the frame
            self.artist.penup()
            self.artist.goto(-50, 0)
            self.artist.pendown()
            self.artist.forward(100)

            # Draw the front wheel
            self.artist.penup()
            self.artist.goto(50, 0)
            self.artist.pendown()
            self.artist.circle(20)

            # Draw the back wheel
            self.artist.penup()
            self.artist.goto(-50, 0)
            self.artist.pendown()
            self.artist.circle(20)

            # Draw the handlebars
            self.artist.penup()
            self.artist.goto(30, 20)
            self.artist.pendown()
            self.artist.left(45)
            self.artist.forward(20)
            self.artist.backward(40)

            # Draw the seat
            self.artist.penup()
            self.artist.goto(0, 20)
            self.artist.pendown()
            self.artist.right(135)
            self.artist.forward(30)



        elif shape == "lol":
            #for _ in range(5):
            self.artist.penup()
            self.artist.right(120)
            self.artist.forward(40)
            self.artist.right(90)
            self.artist.forward(40)
            self.artist.pendown()
            for _ in range(4):
                self.artist.forward(15)
                self.artist.right(90)
            self.artist.right(100)
        elif shape == "hexagon":
            for _ in range(6):
                self.artist.forward(80)
                self.artist.right(60)
        elif shape == "kid":
            # Draw head (a filled circle with a skin tone)
            turtle.penup()
            turtle.goto(0, 50)
            turtle.pendown()
            turtle.color("black", "#FAD6A5")  # outline color and fill color (skin tone)
            turtle.begin_fill()
            turtle.circle(50)
            turtle.end_fill()

            # Draw hair (a filled semicircular arc on top of the head)
            turtle.penup()
            turtle.goto(-50, 100)
            turtle.pendown()
            turtle.color("brown", "brown")
            turtle.begin_fill()
            turtle.setheading(90)  # Point upward
            turtle.circle(50, 180)  # Draw a 180° arc with radius 50
            turtle.goto(-50, 100)  # Close the shape
            turtle.end_fill()

            # Draw eyes (two small filled circles)
            turtle.color("black")
            turtle.penup()
            turtle.goto(-20, 120)
            turtle.pendown()
            turtle.begin_fill()
            turtle.circle(5)
            turtle.end_fill()

            turtle.penup()
            turtle.goto(20, 120)
            turtle.pendown()
            turtle.begin_fill()
            turtle.circle(5)
            turtle.end_fill()

            # Draw mouth (a smiling arc)
            turtle.penup()
            turtle.goto(-20, 80)
            turtle.pendown()
            turtle.setheading(-60)
            turtle.circle(20, 120)  # Arc of 120° with radius 20

            # Draw body (a simple dress represented by an inverted triangle)
            turtle.penup()
            turtle.goto(0, 50)
            turtle.circle(20)

            # Draw arms (two simple lines)
            turtle.pensize(3)
            turtle.color("black")
            turtle.penup()
            turtle.goto(0, 30)
            turtle.pendown()
            turtle.goto(-90, -20)

            turtle.penup()
            turtle.goto(0, 30)
            turtle.pendown()
            turtle.goto(90, -20)

            # Draw legs (two lines from the bottom of the dress)
            turtle.penup()
            turtle.goto(-40, -100)
            turtle.pendown()
            turtle.goto(-40, -200)

            turtle.penup()
            turtle.goto(40, -100)
            turtle.pendown()
            turtle.goto(40, -200)

            # Hide the turtle and finish drawing


            print("I've reached kid")
            x = 10
            for _ in range(475):
                x = x + 5
                turtle.artist.forward(x)
                turtle.artist.right(2 * x)
                turtle.artist.color(random.choice(coolArray))
                #self.artist.color(random.rand(1,2))
        elif shape == "orange":

            turtle.speed(3)
            turtle.width(2)


            turtle.penup()
            turtle.goto(0, -100)  # Move to a position so the circle is centered
            turtle.pendown()
            turtle.color("orange")
            turtle.begin_fill()
            turtle.circle(100)  # Adjust the radius as needed
            turtle.end_fill()

            # Draw the stem
            turtle.penup()
            turtle.goto(0, 0)  # Start at the top of the orange
            turtle.pendown()
            turtle.color("brown")
            turtle.width(5)
            turtle.setheading(90)  # Pointing upward
            turtle.forward(50)

            # Draw a leaf on the stem
            turtle.color("green")
            turtle.goto(0, 100)
            turtle.width(2)
            turtle.begin_fill()
            turtle.right(45)
            turtle.forward(50)
            turtle.right(90)
            turtle.forward(50)
            turtle.right(135)
            turtle.forward(70)  # This brings the turtle back to the leaf's starting point
            turtle.end_fill()
        turtle.update_status(f"Drew a {shape}")

    def process_command(self, command_text):
        """Process a command string"""
        command_text = command_text.lower().strip()

        # Check for exit command
        if "exit" in command_text:
            return False

        # Check for clear command
        if "clear" in command_text:
            self.artist.clear()
            self.artist.penup()
            self.artist.home()
            self.artist.pendown()
            self.update_status("Screen cleared")
            return True

        # Check for draw commands
        if "draw" in command_text:
            for shape in COMMANDS["Shapes"]:
                if shape.lower() in command_text:
                    self.draw_shape(shape)
                    return True
        if "tree" in command_text:
            self.artist.color("green")
            self.artist.begin_fill()
            self.artist.left(45)
            for _ in range(6):
                self.artist.circle(100, 90)
                self.artist.right(45)
                self.artist.circle(100 // 2, 90)
                self.artist.left(45)
            for _ in range(6):


                self.artist.right(45)
                self.artist.circle(100 // 2, 90)
                self.artist.left(45)
            self.artist.end_fill()

            return True
        # Check for color commands
        if "color" in command_text:
            for color in COMMANDS["Colors"]:
                if color.lower() in command_text:
                    self.change_color(color)
                    return True

        # Check for background commands
        if "background" in command_text:
            for color in COMMANDS["Colors"]:
                if color.lower() in command_text:
                    self.change_background(color)
                    return True

        self.update_status("Command not recognized. Please try again.")
        return True

    def change_color(self, color):
        """Change the pen color"""
        color = color.lower()
        if color not in [c.lower() for c in COMMANDS["Colors"]]:
            self.update_status(f"Unknown color. Valid colors: {', '.join(COMMANDS['Colors'])}")
            return
        self.current_color = color
        self.artist.color(color)
        self.update_status(f"Changed color to {color}")

    def change_background(self, color):
        """Change the background color"""
        color = color.lower()
        if color not in [c.lower() for c in COMMANDS["Colors"]]:
            self.update_status(f"Unknown color. Valid colors: {', '.join(COMMANDS['Colors'])}")
            return
        wn.bgcolor(color)
        self.update_status(f"Changed background to {color}")


def record_and_recognize(graphics):
    """Record audio and convert to text using vosk"""
    try:
        # Record audio
        duration = 5  # seconds
        recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype=np.int16)
        graphics.update_status("Listening...")
        sd.wait()

        # Convert audio to text
        recognizer.AcceptWaveform(recording.tobytes())
        result = json.loads(recognizer.Result())

        if result["text"]:
            recognized_text = result["text"]
            graphics.update_history(f"Heard: {recognized_text}", True)
            graphics.update_status(f"Recognized: {recognized_text}")
            return recognized_text
        else:
            graphics.update_history("No speech detected", False)
            return None

    except Exception as e:
        print(f"Error recording: {str(e)}")
        return None


def main():
    # Let user select microphone
    device_id = select_microphone()

    # Initialize graphics
    graphics = TurtleGraphics()
    graphics.update_status("Starting... Say commands clearly")
    print("I'm starting rip program")

    running = True
    while running:
        try:
            # Record and recognize speech
            command_text = record_and_recognize(graphics)

            # Process command if speech was detected
            if command_text:
                running = graphics.process_command(command_text)

        except Exception as e:
            print(f"Error: {str(e)}")
            graphics.update_status(f"Error: {str(e)}")

    print("\nExiting program...")
    wn.bye()


if __name__ == "__main__":
    main()
