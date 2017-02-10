import tkinter
from PIL import Image, ImageTk


def show_top_5(file_names, captions, height=285,
    title="Top 5 results from comparing captions"):
    """
    Takes in file names and captions of 6 images, 1 query and 5 results, and
    creates a Tkinter ui window which displays them.
    """
    # Create window
    Tk = tkinter.Tk()
    Tk.title(title)
    Tk.configure(background='white')
    Tk.wm_geometry("%dx%d%+d%+d" % (1250, 700, 0, 0))

    # Load images
    for i, file_name in enumerate(file_names):
        image = Image.open(file_name)

        # Rescale height
        [image_width, image_height] = image.size
        scale_factor = height / image_height
        new_height = height
        new_width = int(image_width * scale_factor)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # Add border if query image
        if i == 0:
            border_im = Image.new("RGB", ((new_width + 10), (new_height + 10)))
            border_im.paste(image, (5, 5))
            image = border_im

        # Show image
        image = ImageTk.PhotoImage(image)
        label = tkinter.Label(image=image)
        label.image = image
        label.grid(row=(2 * (int(i / 3) + 1)), column=int((i % 3) + 1),
             sticky='nw')

        # Add caption
        if i == 0:
            label=tkinter.Label(text='Query. ' + captions[i]).grid(
                row=(2*(int(i/3)+1)-1),column=int((i%3)+1),sticky='nw')
        else:       
            label=tkinter.Label(text=str(i) + '. ' + captions[i]).grid(
                row=(2*(int(i/3)+1)-1),column=int((i%3)+1),sticky='nw')

    # Load window
    Tk.lift()
    print('Done')
    Tk.mainloop()
