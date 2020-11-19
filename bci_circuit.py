"""BCI Circuit"""

# imports 
import RPi.GPIO as gpio # GPIO = general purpose input/output
import time
import os

red_led_pin = 7 # yellow wire
green_led_pin = 11 # brown wire
buzzer_pin = 13 # black wire

gpio.setmode(gpio.BOARD) # initializes gpio board
gpio.setup(red_led_pin, gpio.OUT) # red light as an output
gpio.setup(green_led_pin, gpio.OUT) # green light as an output
gpio.setup(buzzer_pin, gpio.OUT) # buzzer sound as an output


gpio.output(red_led_pin, gpio.LOW) # turns off red light
gpio.output(green_led_pin, gpio.LOW) # turns off green
gpio.output(buzzer_pin, gpio.HIGH) #turns off buzzer (opposite because using a transistor)

try:
    while True:
        store_val = os.popen("python client.py").read()
        print(store_val)
            val = float(store_val)

        if(val >= 0 and val < 0.7):
            print("feature 0")
                gpio.output(red_led_pin, gpio.LOW) # turns off red light
                gpio.output(green_led_pin, gpio.LOW) # turns off green light
                gpio.output(buzzer_pin, gpio.HIGH) #turns off buzzer (opposite because using a transistor)

        elif(val >= 0.7 and val < 1.5):
            print("feature 1")
            gpio.output(red_led_pin, gpio.HIGH) # turns on red light
            gpio.output(green_led_pin, gpio.LOW) # turns off green light
            gpio.output(buzzer_pin, gpio.HIGH) #turns off buzzer (opposite because using a transistor)

        elif(val >= 1.5 and val < 2.5):
            print("feature 2")
            gpio.output(green_led_pin, gpio.HIGH) # turns on green light
            gpio.output(red_led_pin, gpio.LOW) # turns off red light
            gpio.output(buzzer_pin, gpio.HIGH) #turns off buzzer (opposite because using a transistor)

        elif(val >= 2.5):
            print("feature 3")
            gpio.output(buzzer_pin, gpio.LOW) #turns on buzzer (opposite because using a transistor)
            gpio.output(red_led_pin, gpio.LOW) # turns off red light
            gpio.output(green_led_pin, gpio.LOW) # turns off green light

except KeyboardInterrupt:
    gpio.cleanup()
