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
                 w2v=False,
                 one_motivation=False):
        # Window
        self.root = tkinter.Tk()
        self.root.title("Demo")
        self.root.configure(background='white')
        self.root.wm_geometry("%dx%d%+d%+d" % (1250, 700, 0, 0))
        self.w2v_dic = np.load(path_w2v_dic).item()
        self.one_motivation = one_motivation
        self.w2v_model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin',
                                                           binary=True)
        # self.w2v_model.init_sims(replace=True)  # TODO
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
        # Add line
        entry = tkinter.Entry(self.root, text='Input:')
        entry.place(relx=0.5, rely=0.1)
        entry.bind("<Return>", lambda e: self.find_closest_to_entered_text(entry.get()))
        self.labels.append(entry)

        # Find 5 similar images
        if self.image_list:
            list_images = self.image_list
            captions = [' ', ' ', self.found_vector, ' ', ' ']
        else:
            list_images = self.find_closest_images(query_file_name)
            captions = self.find_explanatory_words(query_file_name, list_images)

        # Load images
        query_size = 450
        for i, file_name in enumerate(list_images):
            image = Image.open(self.path_data + file_name)

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
                name=str(i), image=image,
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

    def find_closest_to_entered_text(self, text):
        vec_resultant = np.zeros((1, 300))
        found_vector = ''
        for w in text.split():
            try:
                vec_resultant += self.w2v_model[w]
                found_vector += w + ' '
            except KeyError:
                pass

        # Top 5 images
        top5_file_names = ['a', 'a', 'a', 'a', 'a']
        top5_distances = [1.0, 1.0, 1.0, 1.0, 1.0]
        # Iterate over all images
        query_w2v = vec_resultant
        for file_name in self.w2v_dic.keys():

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
        self.found_vector = found_vector
        self.image_list = top5_file_names
        self.build_window(self.random_image())

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
    d = Demo(w2v=False, one_motivation=False)
    d.root.mainloop()
