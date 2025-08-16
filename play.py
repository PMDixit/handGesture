import pygame
import os
pygame.mixer.init()
pygame.mixer.music.load(os.path.join("temp","welcome.mp3"))
pygame.mixer.music.play()
while pygame.mixer.music.get_busy(): # check if the file is playing
	pass
pygame.mixer.quit()