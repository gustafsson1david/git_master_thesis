import tkinter
import numpy as np
from PIL import Image, ImageTk
import random
from scipy.spatial.distance import cosine
from gensim.models.keyedvectors import KeyedVectors
import re


class Demo(object):

    def __init__(self,
                 path_w2v_dic='../../data/test_image_space_20170415-1810.npy',
                 path_explanatory_dic='../../data/explanatory_dic.npy',
                 path_data='../../data/val2014/',
                 w2v=False,
                 one_motivation=False):
        # Window
        self.root = tkinter.Tk()
        self.root.title("Demo")
        self.root.configure(background='white')
        self.root.wm_geometry("%dx%d%+d%+d" % (1250, 700, 0, 0))
        self.w2v_dic = np.load(path_w2v_dic).item()
        self.one_motivation = one_motivation
        self.w2v = False
        if w2v:
            self.w2v_model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin',
                                                               binary=True)
            self.w2v_model.init_sims(replace=True)
            self.w2v = True
        self.explanatory_dic = np.load(path_explanatory_dic).item()
        self.path_data = path_data
        self.labels = []
        self.images = []
        start_image = self.random_image()
        self.build_window(start_image)

    def random_image(self):
        # Random select query image and caption
        query_file_name = random.choice(list(self.w2v_dic.keys()))
        return query_file_name

    def build_window(self, query_file_name):
        # Find 5 similar images
        list_images = self.find_closest_images(query_file_name)
        captions = [re.findall('sun\/[a-z_\\\\]+', s)[0][4:] for s in ([query_file_name] + list_images[:-1])]
        new_names = np.load('../../data/new_scene_names.npy').item()
        captions = [new_names[c][0][0] for c in captions]

        # Load images
        query_size = 450
        for i, file_name in enumerate([query_file_name] + list_images[:-1]):
            image = Image.open(self.path_data + file_name)

            # Check if query image
            if i == 0:
                # Resize to query_size x query_size
                # [image_width, image_height] = image.size
                # scale_factor = query_size / image_height
                # new_height = int(image_height * scale_factor)
                # new_width = int(image_width * scale_factor)
                # image = image.resize((new_width, new_height), Image.ANTIALIAS)
                #
                # image = ImageTk.PhotoImage(image)
                # label = tkinter.Label(image=image)
                # label.image = image
                # label.place(
                #     relx=0.5, y=query_size / 2 + 10, anchor=tkinter.CENTER
                # )
                # self.labels.append(label)
                #
                # # Add line
                # line_im = Image.new("RGB", (1250, 2))
                # image = ImageTk.PhotoImage(line_im)
                # label = tkinter.Label(image=image)
                # label.image = image
                # label.place(relx=0.5, y=query_size + 15, anchor=tkinter.CENTER)
                # self.labels.append(label)

                # Add reset button
                button = tkinter.Button(
                    self.root, text="Reset", command=self.reset_image
                )
                button.place(relx=0.9, rely=0.1)
                self.labels.append(button)

            # else:
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
                text=captions[i], compound=tkinter.BOTTOM
            )
            label.image = image
            label.place(
                relx=(1.0 / 6.0) * (i+1), y=query_size + 45 + 185 / 2,
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
                distance = cosine(query_w2v, w2v)

                # See if closest caption is in top 5
                for i, closest_dis in enumerate(top5_distances):
                    if distance < closest_dis:
                        top5_distances[i:i] = [distance]
                        top5_file_names[i:i] = [file_name]
                        del top5_distances[-1]
                        del top5_file_names[-1]
                        break
        return top5_file_names

    def find_explanatory_words(self, query_file_name, list_images):
        # List of words
        query_w2v = self.w2v_dic[query_file_name]
        top_words = []

        if self.w2v:
            if self.one_motivation:
                agg_w2v = query_w2v + sum([self.w2v_dic[f] for f in list_images])
                top_word = self.w2v_model.most_similar([agg_w2v], topn=1, restrict_vocab=30000)[0][0]
                return [' ', ' ', top_word, ' ', ' ']
            else:
                for file_name in list_images:
                    agg_w2v = query_w2v + self.w2v_dic[file_name]
                    top_words.append(self.w2v_model.most_similar([agg_w2v], topn=1, restrict_vocab=30000)[0][0])
                return top_words
        else:
            if self.one_motivation:
                agg_w2v = query_w2v + sum([self.w2v_dic[f] for f in list_images])
                closest_dis = 1.0
                for word in self.explanatory_dic.keys():
                    word_w2v = self.explanatory_dic[word]
                    distance = cosine(agg_w2v, word_w2v)
                    if distance < closest_dis:
                        closest_dis = distance
                        top_word = word
                return [' ', ' ', top_word, ' ', ' ']
            else:
                # Iterate over all images and words
                for file_name in list_images:
                    agg_w2v = query_w2v + self.w2v_dic[file_name]
                    closest_dis = 1.0
                    for word in self.explanatory_dic.keys():
                        word_w2v = self.explanatory_dic[word]
                        distance = cosine(agg_w2v, word_w2v)
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

    def reset_image(self):
        file_name = self.random_image()
        for label in self.labels:
            label.destroy()
        self.labels = []
        self.images = []
        self.build_window(file_name)

if __name__ == "__main__":
    d = Demo(w2v=False, one_motivation=True)
    d.root.mainloop()
