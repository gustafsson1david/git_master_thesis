import tkinter
import numpy as np
from PIL import Image, ImageTk
import random
from scipy.spatial.distance import cosine
from gensim.models.keyedvectors import KeyedVectors


class Demo(object):

    def __init__(self,
                 path_w2v_dic='../../data/image_space_20170413-1646.npy',
                 path_explanatory_dic='../../data/explanatory_dic.npy',
                 path_data='../../data/val2014/',
                 w2v=False):
        # Window
        self.root = tkinter.Tk()
        self.root.title("Demo")
        self.root.configure(background='white')
        # screen size: 1679x1049
        # preferred window size: 1650x1000
        self.root.wm_geometry("%dx%d%+d%+d" % (1650, 1000, 0, 0))
        self.w2v_dic = np.load(path_w2v_dic).item()
        self.w2v_model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin',
                                                           binary=True)
        self.w2v = w2v
        self.explanatory_dic = np.load(path_explanatory_dic).item()
        self.path_data = path_data
        self.labels = []
        self.images = []
        start_image = self.random_image()
        self.image_list = []
        self.found_vector = ' '
        self.build_window(start_image)

    def random_image(self):
        # Random select query image and caption
        query_file_name = random.choice(list(self.w2v_dic.keys()))
        return query_file_name

    def build_window(self, query_file_name):

        for label in self.labels:
            label.destroy()

        # Add textbox
        textbox = tkinter.Entry(self.root, width=50, font=("Segoe UI", 16))
        textbox.place(relx=0.25, y=20, anchor=tkinter.W)
        textbox.bind("<Return>", lambda e: self.text_search(textbox.get()))
        self.labels.append(textbox)

        # Add randomize button
        button = tkinter.Button(self.root, text="Randomize", font=("Segoe UI", 16), command=self.reset_image)
        button.place(relx=0.75, y=20, anchor=tkinter.E)
        self.labels.append(button)

        # Read query image
        image = Image.open(self.path_data + query_file_name)

        # Resize to query_size x query_size
        query_size = 600
        [image_width, image_height] = image.size
        scale_factor = query_size / image_height
        new_height = int(image_height * scale_factor)
        new_width = int(image_width * scale_factor)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # Add image
        image = ImageTk.PhotoImage(image)
        q_image = tkinter.Label(image=image)
        q_image.image = image
        q_image.place(relx=0.5, y=20+20, anchor=tkinter.N)
        self.labels.append(q_image)

        # Add horizontal line
        line_im = ImageTk.PhotoImage(Image.new("RGB", (1250, 2)))
        hline = tkinter.Label(image=line_im)
        hline.image = line_im
        hline.place(relx=0.5, y=20+20+query_size+10, anchor=tkinter.N)
        self.labels.append(hline)

        # Find 5 similar images
        if self.image_list:
            list_images = self.image_list
            self.image_list = []
        else:
            list_images = self.image_search(query_file_name)

        # Add motivation
        word = self.find_word(query_file_name, list_images)
        motivation = tkinter.Label(text='More {} images'.format(word), font=("Segoe UI bold", 16))
        motivation.place(relx=0.5, y=20+20+query_size+10+10, anchor=tkinter.N)
        self.labels.append(motivation)

        # Load images
        for i, file_name in enumerate(list_images):
            image = Image.open(self.path_data + file_name)

            # Resize to y x 100
            [image_width, image_height] = image.size
            scale_factor = 285 / image_height
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
            label = tkinter.Label(name=str(i), image=image, compound=tkinter.BOTTOM)
            label.image = image
            label.place(relx=(1.0/6.0)*(i+1), y=20+20+query_size+10+10+30, anchor=tkinter.N
            )
            self.labels.append(label)

            # Bind click-event
            label.bind("<Button-1>", self.change_image)

    def text_search(self, text):
        vec_resultant = np.zeros((1, 300))
        found_vector = ''
        for w in text.split():
            try:
                vec_resultant += self.w2v_model[w]
                found_vector += w + ' '
            except KeyError:
                pass

        # Top 6 images
        top6_file_names = ['a', 'a', 'a', 'a', 'a', 'a']
        top6_distances = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Iterate over all images
        query_w2v = vec_resultant
        for file_name in self.w2v_dic.keys():

            w2v = self.w2v_dic[file_name]
            # Iterate over all captions belonging to an image
            distance = cosine(query_w2v, w2v)

            # See if closest caption is in top 6
            for i, closest_dis in enumerate(top6_distances):
                if distance < closest_dis:
                    top6_distances[i:i] = [distance]
                    top6_file_names[i:i] = [file_name]
                    del top6_distances[-1]
                    del top6_file_names[-1]
                    break
        print(found_vector)
        self.image_list = top6_file_names[1:]
        self.build_window(top6_file_names[0])

    def image_search(self, query_file_name):
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

    def find_word(self, query_file_name, list_images):
        # List of words
        query_w2v = self.w2v_dic[query_file_name]

        if self.w2v:
            agg_w2v = query_w2v + sum([self.w2v_dic[f] for f in list_images])
            top_word = self.w2v_model.most_similar([agg_w2v], topn=1, restrict_vocab=30000)[0][0]
            return [' ', ' ', top_word, ' ', ' ']
        else:
            agg_w2v = query_w2v + sum([self.w2v_dic[f] for f in list_images])
            closest_dis = 1.0
            for word in self.explanatory_dic.keys():
                word_w2v = self.explanatory_dic[word]
                distance = cosine(agg_w2v, word_w2v)
                if distance < closest_dis:
                    closest_dis = distance
                    top_word = word
            return top_word

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
    d = Demo(w2v=False)
    d.root.mainloop()
