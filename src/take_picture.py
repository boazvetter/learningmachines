import time
import robobo
import cv2

print("Now trying to connect with robobo")
rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.86")
rob.talk("Tilting")
rob.set_phone_tilt(100, 50)
rob.set_phone_tilt(109, 5)
# rob.set_phone_pan(180, 40)
time.sleep(3)

for i in range(0,13):
    rob.talk("Taking a picture")
    img = rob.get_image_front()
    cv2.imwrite('robotviews/robotview{}.jpg'.format(int(round(time.time()))), img)
    time.sleep(1)
    rob.move(31,30,500)
    time.sleep(1)

