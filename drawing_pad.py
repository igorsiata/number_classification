import tkinter as tk
import subprocess
import os
import io
from PIL import Image, ImageGrab
import cv2
import matplotlib.pyplot as plt
import numpy as np
from network import Network


class App(tk.Tk):
    def __init__(self, train_network=True):
        tk.Tk.__init__(self)
        self.title("Number detector")
        self.line_start = None
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.bind('<B1-Motion>', self.draw)

        self.button_clear = tk.Button(self, text="clear",
                                      command=self.clear_canvas)
        self.canvas.pack()
        self.button_clear.pack(pady=10)

        self.predict_frame = tk.Frame(self)
        self.predict_frame.columnconfigure(0, weight=1)
        self.predict_frame.columnconfigure(1, weight=3)

        self.button_predict = tk.Button(self.predict_frame, text="predict",
                                        command=self.predict)
        self.button_predict.grid(row=0, column=0, padx=10)
        self.prediction = tk.Label(
            self.predict_frame, text=f"number: -, certainity = -%")
        self.prediction.grid(row=0, column=1, padx=10)
        self.predict_frame.pack(pady=10)
        self.network = Network([30], True)
        if train_network:
            # can modify those values to change behaviour of network
            self.network.SGD(epochs=30, mini_batch_size=10, eta=3)
        else:
            self.network.load_network()

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_rectangle(x-16, y-16, x+16, y+16, fill='black')

    def predict(self):
        self.save_canvas_as_png()
        img_resizied = self.resize()

        number, probability = self.network.predict(img_resizied)
        self.prediction.config(
            text=f"number: {number}, certainity = {probability*100:.0f}%")
        os.remove("temp.png")
        return number, probability

    def save_canvas_as_png(self):
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + width, y + height)
                       ).save('temp.png', 'PNG')

    def resize(self):
        image = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(image, (28, 28))
        vector = 1 - np.reshape(small, (784,), order='C')/255
        return vector

    def clear_canvas(self):
        self.canvas.delete("all")


if __name__ == "__main__":
    app = App(False)
    print(f"Skuteczność modelu: {app.network.evaluate()*100:.2f}%")
    app.mainloop()
