from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from functools import partial
import random

WIDTH = 500
HEIGHT = 300

root = Tk()
root.title("Captcha")

mainframe = ttk.Frame(root, padding="10", width=WIDTH, height=HEIGHT)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

# List to store the labels
image_labels = [
    "cat",
    "dog",
    "bird",
    "fish",
    "horse",
    "cow",
    "sheep",
    "elephant",
    "lion",
    "tiger",
    "bear",
    "zebra",
    "giraffe",
    "monkey",
    "kangaroo",
    "panda",
    "rabbit",
    "turtle",
    "frog",
    "snake",
    "whale",
    "dolphin",
]

clicked = [False for i in range(9)]

def on_image_click(event, index):
    clicked[index] = not clicked[index]
    label = labels[index]
    if clicked[index]:
        label.config(bg="lightblue")
    else:
        label.config(bg="lightgray")
    print(f"Image {index + 1} clicked!")
    print(clicked)


random_idx = random.sample(range(9), 3)
random_animal = random.choice(image_labels)
other_animals = image_labels.copy()
other_animals.remove(random_animal)
selected_labels = [0 for _ in range(9)]

for i in range(9):
    if i in random_idx:
        selected_labels[i] = random_animal
    else:
        random.choice(image_labels)
        selected_labels[i] = random.choice(other_animals)

labels = []

# Create 9 labels dynamically
for i in range(9):
    label = Label(
        mainframe,
        text=selected_labels[i],
        width=12,
        height=6,
        bg="lightgray",
        relief="raised",
        borderwidth=2,
    )
    row = i // 3
    col = i % 3
    label.grid(row=row+1, column=col, padx=5, pady=5)
    label.bind("<Button-1>", partial(on_image_click, index=i))
    labels.append(label)

instruction = ttk.Label(mainframe, text=f"Click on the {random_animal} images", font=("Arial", 14))
instruction.grid(column=0, row=0, columnspan=3)

def on_sumbit():
    print("Submit button clicked!")
    for i in range(9):
        if clicked[i]:
            print(f"Image {i + 1} selected: {selected_labels[i]}")
        else:
            print(f"Image {i + 1} not selected: {selected_labels[i]}")
    print(f"Selected animal: {random_animal}")
    print(f"Clicked status: {clicked}")
    # Check if the clicked images match the selected animal
    if all(clicked[i] == (selected_labels[i] == random_animal) for i in range(9)):
        print("Correct selection!")
        messagebox.showinfo("Success", "Correct selection!")
    else:
        print("Incorrect selection!")
        messagebox.showerror("Error", "Incorrect selection!")
    # Reset clicked status
    for i in range(9):
        clicked[i] = False
    # Reset the labels

# Submit button
button = ttk.Button(mainframe, text="Submit", command=on_sumbit)
button.grid(column=1, row=4, pady=10)

root.mainloop()
