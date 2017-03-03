import tkinter
import numpy as np
from PIL import Image, ImageTk
import scipy
import random


class demo():

    def __init__(self, path_w2v_dic, path_explanatory_dic, path_data):
        # Window
        self.root = tkinter.Tk()
        self.root.title("Demo")
        self.root.configure(background='white')
        self.root.wm_geometry("%dx%d%+d%+d" % (1250, 700, 0, 0))
        self.w2v_dic = np.load(path_w2v_dic).item()
        self.explanatory_dic = np.load(path_explanatory_dic).item()
        self.path_data = path_data
        self.labels = []
        self.images = []
        start_image = self.random_image()
        self.build_window(start_image)
        self.root.mainloop()

    def random_image(self):
        # Random select query image and caption
        query_file_name = random.choice(list(self.w2v_dic.keys()))
        return query_file_name

    def build_window(self, query_file_name):
        # Find 5 similar images
        list_images = self.find_closest_images(query_file_name)
        captions = self.find_explanatory_words(query_file_name, list_images)

        # Load images
        query_size = 450
        for i, file_name in enumerate([query_file_name] + list_images):
            image = Image.open(self.path_data + file_name)

            # Check if query image
            if i == 0:
                # Resize to query_size x query_size
                [image_width, image_height] = image.size
                scale_factor = query_size / image_height
                new_height = int(image_height * scale_factor)
                new_width = int(image_width * scale_factor)
                image = image.resize((new_width, new_height), Image.ANTIALIAS)

                image = ImageTk.PhotoImage(image)
                label = tkinter.Label(image=image)
                label.image = image
                label.place(
                    relx=0.5, y=query_size / 2 + 10, anchor=tkinter.CENTER
                )
                self.labels.append(label)

                # Add line
                line_im = Image.new("RGB", (1250, 2))
                image = ImageTk.PhotoImage(line_im)
                label = tkinter.Label(image=image)
                label.image = image
                label.place(relx=0.5, y=query_size + 15, anchor=tkinter.CENTER)
                self.labels.append(label)
            else:
                # Resize to y x 100
                [image_width, image_height] = image.size
                scale_factor = 185 / image_height
                new_height = int(image_height * scale_factor)
                new_width = int(image_width * scale_factor)
                image = image.resize((new_width, new_height), Image.ANTIALIAS)

                # Crop to fit
                if new_width > (1240 / 6):
                    top = 0
                    bottom = new_height
                    left = new_width / 2 - 1240 / 12
                    right = new_width / 2 + 1240 / 12
                    image = image.crop((left, top, right, bottom))

                # Place image
                image = ImageTk.PhotoImage(image)
                self.images.append(file_name)
                label = tkinter.Label(
                    name=str(i - 1), image=image,
                    text=captions[i - 1], compound=tkinter.BOTTOM
                )
                label.image = image
                label.place(
                    relx=(1.0 / 6.0) * i, y=query_size + 45 + 185 / 2,
                    anchor=tkinter.CENTER
                )
                self.labels.append(label)

                # Bind click-event
                label.bind("<Button-1>", self.change_image)

    def find_closest_images(self, query_file_name):
        # top 5 images
        top5_file_names = ['a', 'a', 'a', 'a', 'a']
        top5_distances = [1.0, 1.0, 1.0, 1.0, 1.0]

        # Iterate over all images
        query_w2v = self.w2v_dic[query_file_name]
        for file_name in self.w2v_dic.keys():
            if file_name != query_file_name:
                w2v = self.w2v_dic[file_name]
                # Iterate over all captions belonging to an image
                distance = scipy.spatial.distance.cosine(query_w2v, w2v)

                # See if closest caption is in top 5
                for i, closest_dis in enumerate(top5_distances):
                    if distance < closest_dis:
                        top5_distances[i] = distance
                        top5_file_names[i] = file_name
                        break
        return top5_file_names

    def find_explanatory_words(self, query_file_name, list_images):
        # List of words
        query_w2v = self.w2v_dic[query_file_name]
        top_words = []

        # Iterate over all images and words
        for file_name in list_images:
            agg_w2v = query_w2v + self.w2v_dic[file_name]
            closest_dis = 1.0
            for word in self.explanatory_dic.keys():
                word_w2v = self.explanatory_dic[word]
                distance = scipy.spatial.distance.cosine(agg_w2v, word_w2v)
                if distance < closest_dis:
                    closest_dis = distance
                    top_word = word
            top_words.append(top_word)
        return top_words

    def change_image(self, event):
        file_name = self.images[int(str(event.widget)[1:])]
        for label in self.labels:
            label.destroy()
        self.labels = []
        self.images = []
        self.build_window(file_name)








