# Abstract

# Introduction

# State of the Art

Completely Automated Public Turing Test to tell Computers and Humans Apart (CAPTCHA) is a type of challenge–response Turing test used in computing to determine whether the user is human in order to deter bot attacks and spam. The term was coined in 2003 by Luis von Ahn, Manuel Blum, Nicholas J. Hopper, and John Langford. It is a contrived acronym for "Completely Automated Public Turing test to tell Computers and Humans Apart." A historically common type of CAPTCHA (displayed as reCAPTCHA v1) was first invented in 1997 by two groups working in parallel. This form of CAPTCHA requires entering a sequence of letters or numbers from a distorted image. Because the test is administered by a computer, in contrast to the standard Turing test that is administered by a human, CAPTCHAs are sometimes described as reverse Turing tests.

Two widely used CAPTCHA services are Google's reCAPTCHA and the independent hCaptcha. It takes the average person approximately 10 seconds to solve a typical CAPTCHA. With the rising usage of AI, CAPTCHA scams are increasing and may be at risk of being circumvented.

Due to its widespread use and effectiveness in deterring automated bot traffic, there have been many attacks and defenses that have been developed, both of which have been useful in strengthening CAPTCHA.

## Attack Defense scheme

### Attack
Initially, there was no systematic or agreed-upon methodology for the design or evaluation of CAPTCHA. Thus, CAPTCHA tests had varying degrees of effectiveness. For example, CAPTCHA tests that only use fixed-length words were easy to attack using brute-force methods, due to its limited pool of words. As a result of the lack of security standardization, many attacks were effective against these weak CAPTCHA tests.

### Defense
To prevent brute-force or basic algorithms that could pass CAPTCHA tests, minimal changes were implemented in CAPTCHA tests. For instance, in text recognition CAPTCHA tests, modifications like adding noise, backdrops, and crowding characters together (CCT) were effective in mitigating text-based CAPTCHA attacks. Although these were easy to implement, making these modifications did not only make CAPTCHA tests more difficult against automated bots, but against humans as well.

### Attack
In response, many research groups aimed to focus on creating attacks on the modified text-based CAPTCHA tests, and were successful in doing so. The small changes made as a defense were effective only for a short period of time. With significant advancements in Optical Character Recognition (OCR) achieved through Convolutional Neural Network (CNN) and deep learning techniques, text-based CAPTCHA tests became very ineffective against deterring bot traffic. In addition, text-based CAPTCHA tests were not accessible to some visually impaired people, such as those with dyslexia.

### Defense
As an alternative to text-based CAPTCHA, Chew and Tygar spearheaded image-based CAPTCHA tests. The idea was that image and object recognition still remained a much more difficult problem than text-based recognition. In addition, image-based tests may be more accessible to people with visual impairments like dyslexia. Since then, three main domains of image recognition CAPTCHA tests have been developed: image selection, mouse-dragging, and clicking tests.

### Attack
Although image-recognition CAPTCHA tests were more difficult to attack than text-based CAPTCHA tests, neural networks are still able to be trained in cracking the tests. 

### Defense
Tuong, Huong, and Hoang proposed a new CAPTCHA system called zxCAPTCHA. This image-based CAPTCHA test is comprised of 2 layers of security:
1. Text-based layer
There are 3 groups of text, with 3-5 characters each. They are not words, but instead are random strings of characters. Users must be able to identify these characters in the images.
2. Selection-based layer
Users must select images with the text groups and topic image as the background. Thus, users must be able to recognise images in the foreground and background.

### Defense
Personalized CAPTCHA tests have also been explored. Rather than relying on solely the ability to pass the test, which may depend on the difficulty of the test itself, personalized CAPTCHA tests aims to gather clues about the user that may help to differentiate humans from bots.

For example, a hybrid approach could be composed as follows:
1. IP Layer
This first layer extracts information from the IP address.
2. Cookies, Behavioural
This second layer triggers a further CAPTCAH test if the IP address extracted in the previous layer is suspricious or there are cookies missing.

## ResNet
Deep Residual Learning is a technique proposed by Kaiming He et al. to address the difficulty of training very deep neural networks.  
The main idea is the "skip connection", which means that the ResNet model adds the original input to the output of a layer during the forward pass.  
This helps the network learn more effectively and reduces problems like vanishing gradients.

These types of models are often used in visual recognition tasks and will be our main tool of comparison between the other methods of defense against CAPTCHA  bypassing AI.




# Project Description

# Methodology

# Results and Analysis

# Discussion

# Conclusion

# Appendix




yolo 12, yolo 8, resnet18
matrix: accuracy
for each transformation, 5400 images, 90 classes (90 *60)
