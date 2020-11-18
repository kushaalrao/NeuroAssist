"""Full Circuit"""

# imports 
import RPi.GPIO as gpio # GPIO = general purpose input/output
import time
import socket


red_led_pin = 7 # yellow wire
green_led_pin = 11 # brown wire
buzzer_pin = 13 # black wire

### SETUP ###
gpio.setmode(gpio.BOARD) # initializes gpio board
gpio.setup(red_led_pin, gpio.OUT) # red light as an output
gpio.setup(green_led_pin, gpio.OUT) # green light as an output
gpio.setup(buzzer_pin, gpio.OUT) # buzzer sound as an output

gpio.output(red_led_pin, gpio.LOW) # turns off red light
gpio.output(green_led_pin, gpio.LOW) # turns off green light
gpio.output(buzzer_pin, gpio.HIGH) #turns off buzzer (opposite because using a transistor)

try:
	while True:
		store_val = os.popen("python client.py").read()
		print(store_val)
		if(int(store_val[0]) == 0):
            #turn off everything
            gpio.output(red_led_pin, gpio.LOW) # turns off red light
            gpio.output(green_led_pin, gpio.LOW) # turns off green light
            gpio.output(buzzer_pin, gpio.HIGH) #turns off buzzer (opposite because using a transistor)
        elif(int(store_val[0]) == 1):
            #turn on red light
            gpio.output(red_led_pin, gpio.HIGH) # turns on red light
        elif(int(store_val[0]) == 2):
            #turn on green light
            gpio.output(green_led_pin, gpio.HIGH) # turns on green light
        elif(int(store_val[0]) == 3):
            #turn on buzzer
            gpio.output(buzzer_pin, gpio.LOW) #turns on buzzer (opposite because using a transistor)
except KeyboardInterrupt:
		gpio.cleanup()



# time.sleep(5)
# gpio.output(red_led_pin, gpio.HIGH) # turns on red light
# gpio.output(green_led_pin, gpio.HIGH) # turns on green light
# gpio.output(buzzer_pin, gpio.LOW) #turns on buzzer (opposite because using a transistor)
# time.sleep(5)
# gpio.cleanup()
