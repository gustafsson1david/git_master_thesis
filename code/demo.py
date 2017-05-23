import tkinter
import numpy as np
import random
from PIL import Image, ImageTk
from scipy.spatial.distance import cosine
from gensim.models.keyedvectors import KeyedVectors
from AppKit import NSScreen


class Demo(object):

    def __init__(self,
                 data_path='./data/val2014/',
                 timestamp='20170415-1810',
                 whole_w2v=False,
                 compact_window=False):

        # Set up window
        self.root = tkinter.Tk()
        self.root.title("Demo")
        self.root.configure(background='white')
        self.width = int(0.9*NSScreen.mainScreen().frame().size.width)
        if compact_window:
            self.height = 300
        else:
            self.height = int(0.9*NSScreen.mainScreen().frame().size.height)
        self.root.wm_geometry("%dx%d%+d%+d" % (self.width, self.height, 0, 0))

        # Read data
        self.data_path = data_path
        self.image_space = np.load('./runs/'+timestamp+'/image_space_'+timestamp+'.npy').item()
        self.w2v_model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        self.limited_words = np.load('./data/explanatory_dic.npy').item()

        # Parameters and variables
        self.whole_w2v = whole_w2v
        self.compact_window = compact_window
        self.labels = []
        self.images = []

        # Start service
        self.image_search()

    def build_window(self, query_file_name, retrieved_file_names, word):

        # Update window size
        if self.root.winfo_width() > 1:
            self.width = self.root.winfo_width()
            self.height = self.root.winfo_height()

        # Destroy old labels
        for label in self.labels:
            label.destroy()
        self.labels = []

        # Add textbox
        textbox = tkinter.Entry(self.root, width=int(0.04*self.width), font=("Segoe UI", int(0.016*self.height)))
        textbox.place(relx=0.25, rely=0.02, anchor=tkinter.W)
        textbox.bind("<Return>", lambda e: self.text_search(textbox.get()))
        self.labels.append(textbox)

        # Add randomize button
        button = tkinter.Button(self.root, text="Randomize", font=("Segoe UI", int(0.016*self.height)),
                                command=self.image_search)
        button.place(relx=0.75, rely=0.02, anchor=tkinter.E)
        self.labels.append(button)

        # Read query image
        image = Image.open(self.data_path + query_file_name)

        # Resize to query_size x query_size
        query_size = int(0.6 * self.height)
        [image_width, image_height] = image.size
        scale_factor = query_size / image_height
        new_height = int(image_height * scale_factor)
        new_width = int(image_width * scale_factor)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # Add image
        image = ImageTk.PhotoImage(image)
        q_image = tkinter.Label(image=image)
        q_image.image = image
        q_image.place(relx=0.5, rely=0.02+0.02, anchor=tkinter.N)
        self.labels.append(q_image)

        # Add horizontal line
        line_im = ImageTk.PhotoImage(Image.new("RGB", (1250, 2)))
        hline = tkinter.Label(image=line_im)
        hline.image = line_im
        hline.place(relx=0.5, rely=0.02+0.02+0.6+0.01, anchor=tkinter.N)
        self.labels.append(hline)

        # Add motivation
        motivation = tkinter.Label(text='More {} images'.format(word), font=("Segoe UI bold", int(0.016*self.height)))
        motivation.place(relx=0.5, rely=0.02+0.02+0.6+0.01+0.01, anchor=tkinter.N)
        self.labels.append(motivation)

        # Load images
        self.images = []
        for i, file_name in enumerate(retrieved_file_names):
            image = Image.open(self.data_path + file_name)

            # Resize retrieved image
            [image_width, image_height] = image.size
            scale_factor = 0.3 * self.height / image_height
            new_height = int(image_height * scale_factor)
            new_width = int(image_width * scale_factor)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)

            # Crop to fit
            if new_width > (self.width / 5):
                top = 0
                bottom = new_height
                left = new_width / 2 - self.width / 10
                right = new_width / 2 + self.width / 10
                image = image.crop((left, top, right, bottom))

            # Place image
            image = ImageTk.PhotoImage(image)
            self.images.append(file_name)
            label = tkinter.Label(name=str(i), image=image, compound=tkinter.BOTTOM)
            label.image = image
            label.place(relx=(1.0/6.0)*(i+1), rely=0.02+0.02+0.6+0.01+0.01+0.03, anchor=tkinter.N)
            self.labels.append(label)

            # Bind click-event
            label.bind("<Button-1>", self.image_search)

    def build_compact_window(self, query_file_name, retrieved_file_names):

        # Update window size
        if self.root.winfo_width() > 1:
            self.width = self.root.winfo_width()
            self.height = self.root.winfo_height()

        # Destroy old labels
        for label in self.labels:
            label.destroy()
        self.labels = []

        # Add textbox
        textbox = tkinter.Entry(self.root, width=int(0.04*self.width), font=("Segoe UI", int(0.016*self.height)))
        textbox.place(relx=0.25, rely=0.02, anchor=tkinter.W)
        textbox.bind("<Return>", lambda e: self.text_search(textbox.get()))
        self.labels.append(textbox)

        # Add randomize button
        button = tkinter.Button(self.root, text="Randomize", font=("Segoe UI", int(0.016*self.height)),
                                command=self.image_search)
        button.place(relx=0.75, rely=0.02, anchor=tkinter.E)
        self.labels.append(button)

        # Add horizontal line
        line_im = ImageTk.PhotoImage(Image.new("RGB", (1250, 2)))
        hline = tkinter.Label(image=line_im)
        hline.image = line_im
        hline.place(relx=0.5, rely=0.02+0.02+0.01, anchor=tkinter.N)
        self.labels.append(hline)

        # Load images
        self.images = []
        for i, file_name in enumerate([query_file_name] + retrieved_file_names[:4]):
            image = Image.open(self.data_path + file_name)

            # Resize retrieved image
            [image_width, image_height] = image.size
            scale_factor = 0.8 * self.height / image_height
            new_height = int(image_height * scale_factor)
            new_width = int(image_width * scale_factor)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)

            # Crop to fit
            if new_width > (self.width / 5):
                top = 0
                bottom = new_height
                left = new_width / 2 - self.width / 10
                right = new_width / 2 + self.width / 10
                image = image.crop((left, top, right, bottom))

            # Place image
            image = ImageTk.PhotoImage(image)
            self.images.append(file_name)
            label = tkinter.Label(name=str(i), image=image, compound=tkinter.BOTTOM)
            label.image = image
            label.place(relx=(1.0/6.0)*(i+1), rely=0.02+0.02+0.01+0.01+0.03, anchor=tkinter.N)
            self.labels.append(label)

            # Bind click-event
            label.bind("<Button-1>", self.image_search)

    def find_images_by_vector(self, vec):
        # Top 6 images
        top6_file_names = ['a', 'a', 'a', 'a', 'a', 'a']
        top6_distances = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Iterate over all images
        for file_name in self.image_space.keys():

            temp_vec = self.image_space[file_name]
            # Iterate over all captions belonging to an image
            distance = cosine(vec, temp_vec)

            # See if closest caption is in top 6
            for i, closest_dis in enumerate(top6_distances):
                if distance < closest_dis:
                    top6_distances[i:i] = [distance]
                    top6_file_names[i:i] = [file_name]
                    del top6_distances[-1]
                    del top6_file_names[-1]
                    break
        return top6_file_names

    def text_search(self, text):
        # Sum all words in query text
        vec_resultant = np.zeros((1, 300))
        found_vector = ''
        for w in text.split():
            try:
                vec_resultant += self.w2v_model[w]
                found_vector += w + ' '
            except KeyError:
                pass
        print(found_vector)

        # Retrieve images, find word and build window
        retrieved_images = self.find_images_by_vector(vec_resultant)
        word = self.find_word(retrieved_images)
        if self.compact_window:
            self.build_compact_window(retrieved_images[0], retrieved_images[1:])
        else:
            self.build_window(retrieved_images[0], retrieved_images[1:], word)

    def image_search(self, event=None):
        # Pressed image as query if any pressed, else random
        if event:
            query_file_name = self.images[int(str(event.widget)[1:])]
        else:
            query_file_name = random.choice(list(self.image_space.keys()))
        vec = self.image_space[query_file_name]

        # Retrieve images, find word and build window
        retrieved_images = self.find_images_by_vector(vec)
        word = self.find_word(retrieved_images)
        if self.compact_window:
            self.build_compact_window(query_file_name, retrieved_images[1:])
        else:
            self.build_window(query_file_name, retrieved_images[1:], word)

    def find_word(self, file_names):
        # Look through whole word2vec vocab
        if self.whole_w2v:
            agg_w2v = sum([self.image_space[f] for f in file_names])
            top_word = self.w2v_model.most_similar([agg_w2v], topn=1, restrict_vocab=30000)[0][0]
        # Look through limited vocabulary only
        else:
            agg_w2v = sum([self.image_space[f] for f in file_names])
            closest_dis = 100
            for word in self.limited_words.keys():
                word_w2v = self.limited_words[word]
                distance = cosine(agg_w2v, word_w2v)
                if distance < closest_dis:
                    closest_dis = distance
                    top_word = word
        return top_word

if __name__ == "__main__":
    d = Demo(data_path='./data/val2014/', timestamp='20170415-1810', whole_w2v=False, compact_window=False)
    d.root.mainloop()
